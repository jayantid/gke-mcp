import subprocess
import os
import pytest
import yaml
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from langchain_google_genai import ChatGoogleGenerativeAI
from deepeval.models.base_model import DeepEvalBaseLLM

# Custom wrapper for DeepEval to use LangChain's ChatGoogleGenerativeAI
class GoogleGeminiAI(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = chat_model.invoke(prompt)
        return res.content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "gemini-2.5-flash"

def test_demo_failure():
    """This test is designed to fail to demonstrate DeepEval's error reporting."""
    prompt = "Help me generate a Kubernetes manifest for an Nginx deployment."
    
    # Path to the Go CLI wrapper relative to the repo root
    cli_path = "./cmd/manifestgen_cli/manifestgen_cli"
    
    try:
        # Running the CLI from the repo root
        result = subprocess.run([cli_path, prompt], capture_output=True, text=True, check=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        pytest.fail(f"CLI failed: {e}")

    # Cleanup markdown code blocks if present
    cleaned_output = output.strip()
    if cleaned_output.startswith("```yaml"):
        cleaned_output = cleaned_output[len("```yaml"):]
    if cleaned_output.endswith("```"):
        cleaned_output = cleaned_output[:-len("```")]
    cleaned_output = cleaned_output.strip()

    # DeepEval check: Use GEval with Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.fail("GEMINI_API_KEY environment variable not set")

    chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    gemini_ai_model = GoogleGeminiAI(model=chat_model)

    # STRICT METRIC DESIGNED TO FAIL
    # We ask for Nginx (which defaults to port 80), but strictly require port 8080 in the test.
    failure_metric = GEval(
        name="Strict Port 8080 Check",
        criteria="The output is a valid Kubernetes manifest in YAML format. The container must be exposed on port 8080. If it uses port 80 or any other port, it must FAIL.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=gemini_ai_model
    )

    test_case = LLMTestCase(
        input=prompt,
        actual_output=cleaned_output
    )

    print("\n--- DeepEval will now evaluate a deliberately mismatched case ---")
    print("Expect a failure with a detailed rationale explaining why port 80 was used instead of 8080.")
    
    assert_test(test_case, [failure_metric])
