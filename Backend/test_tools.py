import requests
import json

BASE_URL = "http://localhost:8000"

def test_tools():
    print("Testing AI Tools Endpoints...")
    
    # Test 1: Generate Content (Essay Writer)
    print("\n1. Testing Essay Writer...")
    try:
        payload = {
            "tool": "generator",
            "sub_tool": "essay_writer",
            "topic": "The importance of artificial intelligence in education",
            "content": "The importance of artificial intelligence in education"
        }
        response = requests.post(f"{BASE_URL}/tools/generate", json=payload)
        if response.status_code == 200:
            print("✅ Essay Writer Success")
            # print(response.json()['result'][:100] + "...")
        else:
            print(f"❌ Essay Writer Failed: {response.text}")
    except Exception as e:
        print(f"❌ Essay Writer Error: {str(e)}")

    # Test 2: Math Solver
    print("\n2. Testing Math Solver...")
    try:
        payload = {
            "tool": "solver",
            "sub_tool": "math",
            "content": "Solve x^2 + 5x + 6 = 0"
        }
        response = requests.post(f"{BASE_URL}/tools/solve", json=payload)
        if response.status_code == 200:
            print("✅ Math Solver Success")
            # print(response.json()['result'][:100] + "...")
        else:
            print(f"❌ Math Solver Failed: {response.text}")
    except Exception as e:
        print(f"❌ Math Solver Error: {str(e)}")

    # Test 3: Quiz Generator
    print("\n3. Testing Quiz Generator...")
    try:
        payload = {
            "tool": "generator",
            "sub_tool": "quiz",
            "topic": "Python Programming",
            "content": "Python Programming"
        }
        response = requests.post(f"{BASE_URL}/tools/generate", json=payload)
        if response.status_code == 200:
            print("✅ Quiz Generator Success")
        else:
            print(f"❌ Quiz Generator Failed: {response.text}")
    except Exception as e:
        print(f"❌ Quiz Generator Error: {str(e)}")

if __name__ == "__main__":
    test_tools()
