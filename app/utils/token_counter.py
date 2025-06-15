import tiktoken                       # pip install tiktoken
ENC = tiktoken.get_encoding("cl100k_base")   # good default for Mistral

def count_tokens(text: str) -> int:
    return len(ENC.encode(text))