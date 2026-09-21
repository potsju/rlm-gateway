"""
Workaround for a bug in rlms==0.1.3's AnthropicClient: it assumes
response.content[0] is always a text block, but Claude returns a
ThinkingBlock first when extended thinking is on, which has no .text
attribute. Patches completion()/acompletion() to pick out the first
real text block instead. Safe to delete once upstream fixes this.
"""

from rlm.clients import anthropic as rlm_anthropic


def _first_text(content) -> str:
    for block in content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


def completion(self, prompt, model=None):
    messages, system = self._prepare_messages(prompt)
    model = model or self.model_name
    if not model:
        raise ValueError("Model name is required for Anthropic client.")
    kwargs = {"model": model, "max_tokens": self.max_tokens, "messages": messages}
    if system:
        kwargs["system"] = system
    response = self.client.messages.create(**kwargs)
    self._track_cost(response, model)
    return _first_text(response.content)


async def acompletion(self, prompt, model=None):
    messages, system = self._prepare_messages(prompt)
    model = model or self.model_name
    if not model:
        raise ValueError("Model name is required for Anthropic client.")
    kwargs = {"model": model, "max_tokens": self.max_tokens, "messages": messages}
    if system:
        kwargs["system"] = system
    response = await self.async_client.messages.create(**kwargs)
    self._track_cost(response, model)
    return _first_text(response.content)


rlm_anthropic.AnthropicClient.completion = completion
rlm_anthropic.AnthropicClient.acompletion = acompletion
