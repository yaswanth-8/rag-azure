# azure_openai_check.py
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

def verify_azure_openai_credentials():
    load_dotenv()
    
    # Get credentials from environment
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    
    # Verify environment variables exist
    print("\nChecking Azure OpenAI credentials:")
    missing_vars = []
    if not endpoint:
        missing_vars.append("AZURE_OPENAI_ENDPOINT")
    if not api_key:
        missing_vars.append("AZURE_OPENAI_API_KEY")
    if not deployment_name:
        missing_vars.append("AZURE_DEPLOYMENT_NAME")
    if not api_version:
        missing_vars.append("AZURE_OPENAI_API_VERSION")
    
    if missing_vars:
        print("❌ Missing environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        return False
    
    # Verify endpoint format
    if not endpoint.startswith("https://") or not endpoint.endswith(".openai.azure.com/"):
        print("❌ Invalid endpoint format")
        print("Expected format: https://YOUR_RESOURCE_NAME.openai.azure.com/")
        print(f"Current value: {endpoint}")
        return False
    
    try:
        # Create Azure OpenAI client
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
        
        # Test the connection with a simple completion
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        
        print("✅ Successfully connected to Azure OpenAI")
        print(f"✅ Deployment '{deployment_name}' is working")
        return True
        
    except Exception as e:
        print(f"\n❌ Error connecting to Azure OpenAI:")
        print(f"Error details: {str(e)}")
        return False

if __name__ == "__main__":
    print("Verifying Azure OpenAI credentials...")
    verify_azure_openai_credentials()