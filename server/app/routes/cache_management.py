from fastapi import APIRouter, HTTPException
from app.models.schemas import URLLookupRequest, URLLookupResponse, URLDeleteRequest, URLDeleteResponse
from app.helpers.cache_manager import MAPPING_FILE, get_current_mapping, update_mapping
import pickle
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/lookup", response_model=URLLookupResponse)
async def lookup_url_record(req: URLLookupRequest):
    """
    Lookup a cached record by URL
    """
    try:
        url = req.url.strip()
        
        if not url:
            return URLLookupResponse(
                found=False,
                message="URL cannot be empty"
            )
        
        # Get the current mapping
        current_mapping = get_current_mapping()
        
        # Check if URL exists
        if url in current_mapping:
            record_path = current_mapping[url]
            return URLLookupResponse(
                found=True,
                record=record_path,
                message=f"Record found for URL: {url}"
            )
        else:
            return URLLookupResponse(
                found=False,
                message=f"No record found for URL: {url}"
            )
            
    except Exception as e:
        logger.error(f"Error looking up URL {req.url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/delete", response_model=URLDeleteResponse)
async def delete_url_record(req: URLDeleteRequest):
    """
    Delete a cached record by URL
    """
    try:
        url = req.url.strip()
        
        if not url:
            return URLDeleteResponse(
                success=False,
                message="URL cannot be empty"
            )
        
        # Get the current mapping
        current_mapping = get_current_mapping()
        
        # Check if URL exists
        if url not in current_mapping:
            return URLDeleteResponse(
                success=False,
                message=f"No record found for URL: {url}"
            )
        
        # Get the record path before deletion
        record_path = current_mapping[url]
        
        # Delete the actual vector store files
        try:
            if os.path.exists(record_path):
                os.remove(record_path)
                logger.info(f"Deleted vector store file: {record_path}")
            
            # Also try to remove the directory if it's empty
            dir_path = os.path.dirname(record_path)
            if os.path.exists(dir_path) and not os.listdir(dir_path):
                os.rmdir(dir_path)
                logger.info(f"Deleted empty directory: {dir_path}")
                
        except Exception as e:
            logger.warning(f"Could not delete vector store file {record_path}: {str(e)}")
        
        # Remove from mapping
        del current_mapping[url]
        
        # Update the mapping
        update_mapping(current_mapping)
        
        logger.info(f"Successfully deleted record for URL: {url}")
        
        return URLDeleteResponse(
            success=True,
            message=f"Successfully deleted record for URL: {url}",
            deleted_record=record_path
        )
            
    except Exception as e:
        logger.error(f"Error deleting URL {req.url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/list")
async def list_all_cached_urls():
    """
    List all cached URLs and their record paths
    """
    try:
        # Get the current mapping
        current_mapping = get_current_mapping()
        
        return {
            "total_records": len(current_mapping),
            "records": [
                {
                    "url": url,
                    "path": path,
                    "file_exists": os.path.exists(path)
                }
                for url, path in current_mapping.items()
            ]
        }
            
    except Exception as e:
        logger.error(f"Error listing cached URLs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")