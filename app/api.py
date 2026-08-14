"""
app/api.py — LexiCare FastAPI Production Service.

Features:
  - POST /chat — Legal QA with safety guard + RAG + hallucination check
  - POST /chat/stream — Streaming response via SSE
  - GET /health — Health check + GPU info
  - GET /models — List available model variants

Usage:
    uvicorn app.api:app --host 0.0.0.0 --port 8000
    python -m app.api
"""

import gc
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.domain_config import get_domain

# ──────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────


class ChatRequest(BaseModel):
    query: str = Field(..., description="User's legal question")
    model_variant: str = Field(default="sft", description="Model variant: base, sft, sft+dpo, grpo")
    temperature: float = Field(default=0.7, ge=0.1, le=1.5)
    max_tokens: int = Field(default=400, ge=50, le=1000)
    enable_rag: bool = Field(default=True)
    enable_safety: bool = Field(default=True)
    enable_stream: bool = Field(default=False)
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    query: str
    response: str
    model_variant: str
    intent: str
    confidence: float
    safety_status: str
    rag_used: bool
    rag_docs_count: int
    hallucination_flags: int
    tokens_generated: int
    time_sec: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    gpu_available: bool
    gpu_name: Optional[str]
    vram_used_gb: Optional[float]
    vram_total_gb: Optional[float]
    uptime_sec: float


# ──────────────────────────────────────────────
# Model Manager
# ──────────────────────────────────────────────

class ModelManager:
    """Manages multiple LoRA variants for inference."""

    def __init__(self, base_model: str, lora_paths: dict = None):
        self.base_model_name = base_model
        self.lora_paths = lora_paths or {}
        self.tokenizer = None
        self.models = {}
        self.start_time = time.time()

    def load(self):
        """Load tokenizer (model lazy-loaded on first request)."""
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_model(self, variant: str):
        """Get or load a model variant."""
        if variant in self.models:
            return self.models[variant]

        from transformers import AutoModelForCausalLM
        from peft import PeftModel

        self._clear_gpu()

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        if variant != "base" and variant in self.lora_paths:
            model = PeftModel.from_pretrained(model, self.lora_paths[variant])
            model = model.merge_and_unload()

        model.eval()
        self.models[variant] = model
        return model

    def generate(self, query: str, variant: str = "sft",
                 system_prompt: str = None, max_tokens: int = 400,
                 temperature: float = 0.7) -> tuple:
        """Generate response. Returns (text, num_tokens, time_sec)."""
        from app.intent_router import IntentRouter

        router = IntentRouter()
        decision = router.route(query)
        prompt_text = system_prompt or decision.system_prompt_override or get_domain().default_system_prompt

        formatted = (
            f"<|im_start|>system\n{prompt_text}<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        model = self.get_model(variant)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(model.device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        num_tokens = len(generated)
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        return text, num_tokens, elapsed

    def generate_stream(self, query: str, variant: str = "sft",
                        system_prompt: str = None, max_tokens: int = 400,
                        temperature: float = 0.7):
        """Streaming generator for SSE."""
        from app.intent_router import IntentRouter

        router = IntentRouter()
        decision = router.route(query)
        prompt_text = system_prompt or decision.system_prompt_override or get_domain().default_system_prompt

        formatted = (
            f"<|im_start|>system\n{prompt_text}<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        model = self.get_model(variant)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(model.device)

        from transformers import TextStreamer

        class SimpleStreamer(TextStreamer):
            def __init__(self, tokenizer):
                super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)

            def on_finalized_text(self, text: str, stream_end: bool = False):
                if text.strip():
                    yield text

        # Fallback: generate in chunks (TextStreamer not async-compatible in simple setup)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        # Stream character by character (simple SSE)
        for i in range(0, len(text), 5):
            chunk = text[i:i+5]
            yield f"data: {json.dumps({'text': chunk})}\n\n"
            time.sleep(0.01)

        yield "data: [DONE]\n\n"

    def _clear_gpu(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def gpu_info(self) -> dict:
        if torch.cuda.is_available():
            return {
                "gpu_available": True,
                "gpu_name": torch.cuda.get_device_name(0),
                "vram_used_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
                "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 0),
            }
        return {"gpu_available": False, "gpu_name": None, "vram_used_gb": None, "vram_total_gb": None}


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────

model_manager: Optional[ModelManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown."""
    global model_manager

    base_model = os.environ.get("LEXICARE_MODEL", "Qwen/Qwen3-8B")

    # Auto-detect LoRA adapters
    lora_paths = {}
    lora_dir = Path("outputs/lora_weights")
    if lora_dir.exists():
        for variant, pattern in [("sft", "lexicare-sft-*"),
                                   ("sft+dpo", "lexicare-safedpo-*"),
                                   ("grpo", "lexicare-grpo-*")]:
            candidates = sorted(lora_dir.glob(pattern))
            if candidates:
                lora_paths[variant] = str(candidates[-1])

    try:
        model_manager = ModelManager(base_model, lora_paths)
        model_manager.load()
        print(f"✅ Model loaded: {base_model}")
        print(f"   Available variants: base, {', '.join(lora_paths.keys())}")
    except Exception as e:
        print(f"⚠ Model not loaded (CPU-only or missing model): {e}")
        model_manager = None

    yield

    if model_manager:
        model_manager._clear_gpu()


app = FastAPI(
    title="LexiCare API",
    description="法律咨询智能助手 API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check + GPU info."""
    gpu = model_manager.gpu_info if model_manager else {"gpu_available": False}
    return HealthResponse(
        status="healthy" if model_manager else "degraded",
        model_loaded=model_manager is not None,
        model_name=model_manager.base_model_name if model_manager else "N/A",
        gpu_available=gpu["gpu_available"],
        gpu_name=gpu["gpu_name"],
        vram_used_gb=gpu["vram_used_gb"],
        vram_total_gb=gpu["vram_total_gb"],
        uptime_sec=round(time.time() - (model_manager.start_time if model_manager else time.time()), 1),
    )


@app.get("/models")
async def list_models():
    """List available model variants."""
    variants = ["base"]
    if model_manager and model_manager.lora_paths:
        variants.extend(model_manager.lora_paths.keys())
    return {"variants": variants, "base_model": model_manager.base_model_name if model_manager else "N/A"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Legal QA endpoint with full safety pipeline."""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded. GPU required for inference.")

    from app.safety_guard import SafetyGuard
    from app.intent_router import IntentRouter
    from app.hallucination_detector import HallucinationDetector

    guard = SafetyGuard()
    router = IntentRouter()
    detector = HallucinationDetector()

    # Input check
    input_result = guard.check_input(request.query)
    if not input_result.safe:
        return ChatResponse(
            query=request.query,
            response=input_result.fallback_response,
            model_variant=request.model_variant,
            intent="blocked",
            confidence=1.0,
            safety_status="blocked",
            rag_used=False,
            rag_docs_count=0,
            hallucination_flags=0,
            tokens_generated=0,
            time_sec=0,
            timestamp=datetime.now().isoformat(),
        )

    # RAG
    rag_docs = []
    if request.enable_rag:
        decision = router.route(request.query)
        if decision.needs_rag:
            try:
                from app.rag_retriever import get_retriever
                retriever = get_retriever()
                rag_docs = retriever.retrieve(
                    decision.rag_query if decision.rag_query else request.query,
                    top_k=3,
                )
            except Exception:
                pass

    # Generate
    response_text, num_tokens, gen_time = model_manager.generate(
        request.query,
        variant=request.model_variant,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    # Output check
    output_result = guard.check_output(
        request.query, response_text,
        category=input_result.category,
        retrieved_docs=rag_docs,
    )
    response_text = output_result.response
    confidence = output_result.confidence

    # Hallucination check
    hallu_report = detector.check(response_text, rag_docs, question=request.query)
    hallu_flags = hallu_report.total_count

    return ChatResponse(
        query=request.query,
        response=response_text,
        model_variant=request.model_variant,
        intent=input_result.category,
        confidence=round(confidence, 2),
        safety_status="safe" if output_result.safe else "warning",
        rag_used=len(rag_docs) > 0,
        rag_docs_count=len(rag_docs),
        hallucination_flags=hallu_flags,
        tokens_generated=num_tokens,
        time_sec=round(gen_time, 2),
        timestamp=datetime.now().isoformat(),
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming legal QA endpoint."""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    def generate():
        for chunk in model_manager.generate_stream(
            request.query,
            variant=request.model_variant,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
