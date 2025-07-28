from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class NodeData(BaseModel):
    """Base node data structure"""
    label: str
    nodeId: int
    nodeType: str

class CustomInputData(NodeData):
    """Custom input node data"""
    query: str

class ToolNodeData(NodeData):
    """Tool node data"""
    availableTools: List[str]
    selectedTool: str
    toolActions: str
    toolApiKey: str = ""
    accountLinkRequired: bool = False
    connections: Dict[str, Any] = {}
    execution_metadata: Dict[str, Any] = {}

class LLMNodeData(NodeData):
    """LLM node data"""
    modelProvider: str
    modelName: str = ""
    systemPrompt: str
    apiKey: str
    temperature: float = 0.3

class CustomOutputData(NodeData):
    """Custom output node data"""
    agentOutput: str = ""

class Position(BaseModel):
    """Node position"""
    x: float
    y: float

class WorkflowNode(BaseModel):
    """Workflow node structure"""
    id: str
    type: str
    position: Position
    data: Dict[str, Any]  # Will contain one of the above data types

class WorkflowEdge(BaseModel):
    """Workflow edge structure"""
    id: str
    source: str
    target: str
    sourceHandle: str = "output"
    targetHandle: str = "input"
    type: str = "default"

class WorkflowRequest(BaseModel):
    """Complete workflow execution request"""
    nodes: List[WorkflowNode]
    edges: List[WorkflowEdge]

class WorkflowResponse(BaseModel):
    """Workflow execution response"""
    success: bool
    result: Dict[str, Any]
    execution_id: Optional[str] = None
    error: Optional[str] = None