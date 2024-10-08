cd frontend
python -m http.server 8080 - to start frontend

uvicorn main:app --reload - to start backend

at end- verify
lsof -i :8000 - to see the processes running on these ports

kill -9 <pID> - to kill the running process

----

run azure_credentials_check to check if creds are correct
run azure_openai_check to check if openAI is configured


--- 

run setup_index to setup index