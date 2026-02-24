## Endpoints
Chunks
- X GET /chunks - List all chunks
- X GET /chunks/{file_id} - List chunks for a specific file

Files (Documents & Images)
- X GET /file - List files
- X POST /file/upload - Upload file 
- POST /file/upload/multiple - Upload multiple files 
- X DELETE /file/{file_id} - Delete file with chunks 

# Chat 
- X GET /chat - List all chats
- X GET /chat/{chat_id} - List messages for a specific chat (if query params include with_chunks=true, include the chunks in the response)
- X POST /chat - Send message new chat (if query params include with_chunks=true, include the chunks in the response)
- X POST /chat/{chat_id} - Send message existing chat
- X DELETE /chat/{chat_id} - Delete a chat
- X DELETE /chat - Delete every chat

## User Stories
- Should be able to see RAGAS test metrics per message and a summary of all messages
- Should be able to swap between OpenAI (4o), Antropic (Sonnet-4.6) and Google (Gemini-3-Flash) for image understanding/agent model
- Should be able to create chats with messages per user_id which is fake for now, they get their titles from llm summaries of the query. (Can delete chats)
