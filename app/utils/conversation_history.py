"""
Utility for managing conversation history in memory.
"""
from typing import List, Dict, Optional
from datetime import datetime

class ConversationHistory:
    """
    Manages in-memory conversation history for recent turns with automatic capping.
    """
    _instance = None
    
    def __init__(self, max_turns: int = 10):
        """
        Initialize the conversation history with a maximum number of turns.
        
        Args:
            max_turns: Maximum number of turns to keep in memory
        """
        self.max_turns = max_turns
        self.turns = []
        
    def add_turn(self, user_message: str, assistant_message: str, memory_id: Optional[str] = None) -> None:
        """
        Add a new conversation turn to the history.
        Automatically removes oldest turns if max_turns is exceeded.
        
        Args:
            user_message: The user's message
            assistant_message: The assistant's response
            memory_id: Optional ID of the memory node if stored in Neo4j
        """
        # Add new turn
        self.turns.append({
            'user': user_message,
            'assistant': assistant_message,
            'id': memory_id,
            'timestamp': datetime.now()
        })
        
        # Remove oldest turns if we exceed max_turns
        while len(self.turns) > self.max_turns:
            self.turns.pop(0)
            
    def get_recent_turns(self, limit: int = None) -> List[Dict[str, str]]:
        """
        Get recent conversation turns.
        
        Args:
            limit: Maximum number of turns to return. If None, returns all turns.
            
        Returns:
            List of conversation turns, each containing 'user' and 'assistant' messages
        """
        if limit is None:
            return self.turns
        return self.turns[-limit:]
        
    def clear(self):
        """Clear all conversation history."""
        self.turns = [] 

    @classmethod
    def get_instance(cls, max_turns: int = 10):
        if cls._instance is None:
            cls._instance = cls(max_turns=max_turns)
        return cls._instance