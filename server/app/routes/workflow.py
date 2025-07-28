from fastapi import APIRouter, HTTPException
from app.models.workflow import WorkflowRequest, WorkflowResponse
from app.services.executor import execute_workflow
import logging
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workflow",
    tags=["Workflow Execution"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"}
    }
)

@router.post("/execute", response_model=WorkflowResponse)
async def execute_workflow_endpoint(workflow: WorkflowRequest):
    """
    Execute a workflow with the provided nodes and edges.
    
    This endpoint processes a complete workflow by:
    1. Validating the workflow structure
    2. Executing nodes in topological order
    3. Handling parallel execution where needed
    4. Returning comprehensive results
    
    Args:
        workflow: WorkflowRequest containing nodes and edges
        
    Returns:
        WorkflowResponse with execution results
    """
    execution_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Starting workflow execution {execution_id}")
        logger.info(f"Workflow contains {len(workflow.nodes)} nodes and {len(workflow.edges)} edges")
        
        # Convert Pydantic model to dict for backward compatibility
        workflow_dict = {
            "nodes": [node.dict() for node in workflow.nodes],
            "edges": [edge.dict() for edge in workflow.edges]
        }
        
        # Execute the workflow
        result = await execute_workflow(workflow_dict)
        
        logger.info(f"Workflow execution {execution_id} completed successfully")
        
        return WorkflowResponse(
            success=True,
            result=result,
            execution_id=execution_id
        )
        
    except ValueError as ve:
        logger.error(f"Validation error in workflow {execution_id}: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Workflow validation error: {str(ve)}")
        
    except Exception as e:
        logger.error(f"Error executing workflow {execution_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Health check endpoint for the workflow service.
    """
    return {
        "status": "healthy",
        "service": "workflow-executor",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.post("/validate", response_model=dict)
async def validate_workflow(workflow: WorkflowRequest):
    """
    Validate a workflow structure without executing it.
    
    Args:
        workflow: WorkflowRequest to validate
        
    Returns:
        Validation results
    """
    try:
        from app.services.validator import validate_workflow_structure
        
        workflow_dict = {
            "nodes": [node.dict() for node in workflow.nodes],
            "edges": [edge.dict() for edge in workflow.edges]
        }
        
        validation_result = validate_workflow_structure(workflow_dict)
        
        return {
            "valid": validation_result["valid"],
            "errors": validation_result.get("errors", []),
            "warnings": validation_result.get("warnings", [])
        }
        
    except Exception as e:
        logger.error(f"Error validating workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")