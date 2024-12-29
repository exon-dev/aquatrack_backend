from fastapi import FastAPI, Request, Response

app = FastAPI()

origins = [
 '*'
]

# app.add_middleware(
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

@app.get('/api/v1/test')
async def test(request: Request):
    return Response(content='{"message": "Hello World"}', media_type='application/json')