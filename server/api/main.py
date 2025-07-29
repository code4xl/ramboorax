from app.main import app

# Export the FastAPI app for Vercel
def handler(request):
    return app

# For Vercel serverless function
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)