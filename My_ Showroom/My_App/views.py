from django.shortcuts import render
from django.http import JsonResponse
import sys, os

# Add project root (where app.py lives) to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import generate_response   # now works ✅

# Create your views here.
def index(request):
    return render(request, "index.html")

def chat(request):
    if request.method == "POST":
        question = request.POST.get("question", "")
        print("My Question is", question)
        answer = generate_response(question)
        print(answer)
        return JsonResponse({"response": answer})
    return JsonResponse({"error": "Invalid request"}, status=400)