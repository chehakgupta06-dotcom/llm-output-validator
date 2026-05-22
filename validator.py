import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ── 1. Define your schema with Pydantic ──────────────────────────────────────
class ProductReview(BaseModel):
    product_name: str = Field(description="Name of the product")
    rating: float = Field(description="Rating from 1.0 to 5.0", ge=1.0, le=5.0)
    sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")
    pros: list[str] = Field(description="List of positive points")
    cons: list[str] = Field(description="List of negative points")
    summary: str = Field(description="One sentence summary of the review")


# ── 2. Build the validator with retry logic ───────────────────────────────────
class LLMOutputValidator:
    def __init__(self, max_retries: int = 3):
        self.llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    groq_api_key="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"
)
        self.parser = PydanticOutputParser(pydantic_object=ProductReview)
        self.max_retries = max_retries

        self.prompt = PromptTemplate(
            template="""You are a product review analyst.
Analyze the following product review and extract structured information.

Review:
{review}

{format_instructions}

Return ONLY valid JSON. No markdown, no explanation.""",
            input_variables=["review"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            }
        )

    def validate(self, review_text: str) -> dict:
        attempt = 0
        last_error = None
        raw_outputs = []

        while attempt < self.max_retries:
            attempt += 1
            print(f"\n🔄 Attempt {attempt}/{self.max_retries}")

            try:
                # Call LLM
                chain = self.prompt | self.llm | self.parser
                result = chain.invoke({"review": review_text})

                return {
                    "success": True,
                    "attempts": attempt,
                    "data": result.model_dump(),
                    "raw_outputs": raw_outputs
                }

            except (ValidationError, json.JSONDecodeError, Exception) as e:
                last_error = str(e)
                raw_outputs.append(f"Attempt {attempt} failed: {last_error}")
                print(f"❌ Attempt {attempt} failed: {last_error}")

        return {
            "success": False,
            "attempts": attempt,
            "error": last_error,
            "raw_outputs": raw_outputs
        }


# ── 3. Quick test (run directly) ──────────────────────────────────────────────
if __name__ == "__main__":
    validator = LLMOutputValidator(max_retries=3)

    sample_review = """
    I bought the Sony WH-1000XM5 headphones last month. The noise cancellation 
    is absolutely incredible — I can't hear anything on flights. Sound quality 
    is rich and detailed. Battery lasts about 28 hours which is great. 
    However, the ear cups get uncomfortable after 2-3 hours, and at ₹29,000 
    they're quite expensive. The touch controls are sometimes finicky too.
    """

    result = validator.validate(sample_review)

    if result["success"]:
        print("\n✅ Validation successful!")
        print(f"Attempts needed: {result['attempts']}")
        print(json.dumps(result["data"], indent=2))
    else:
        print(f"\n💀 Failed after {result['attempts']} attempts")
        print(f"Error: {result['error']}")