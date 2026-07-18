# AI Learning Assistant

An intelligent learning assistant that helps students study by uploading PDF materials, asking questions, and tracking their learning progress using AI-powered analysis.

## Features

- 📚 **PDF Upload & Processing**: Upload study materials and automatically extract text chunks
- 🤖 **AI-Powered Q&A**: Ask questions about your uploaded materials with difficulty-based responses
- 📊 **Progress Tracking**: Monitor your learning progress across different topics
- 🧠 **Forgetting Curve**: Get intelligent revision suggestions based on spaced repetition
- 📈 **Mastery Assessment**: Track your mastery levels for different subjects
- 🎯 **Misconception Detection**: Identify and correct misunderstandings in your explanations

## Supported Topics

- Data Structures & Algorithms (DSA)
- Operating Systems (OS)
- Database Management Systems (DBMS)
- Computer Networks (CN)
- Object-Oriented Programming (OOP)

## Tech Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **Google Gemini AI**: Advanced language model for intelligent responses
- **Sentence Transformers**: Semantic similarity for relevant content retrieval
- **PDFPlumber**: PDF text extraction
- **Scikit-learn**: Machine learning for mastery prediction

### Frontend
- **HTML5/CSS3/JavaScript**: Modern web technologies
- **Tailwind CSS**: Utility-first CSS framework
- **Font Awesome**: Beautiful icons

## Installation

### Prerequisites
- Python 3.8+
- Google Gemini API Key

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd SDP/Backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Unix/MacOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create a .env file in the Backend directory
   echo "GEMINI_API_KEY=your_actual_api_key_here" > .env
   ```
   
   Or get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

5. **Create necessary directories**
   ```bash
   mkdir -p data/uploads data/extracted_text
   ```

6. **Run the backend server**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd ../Frontend
   ```

2. **Serve the frontend**
   ```bash
   # Using Python's built-in server
   python -m http.server 3000
   
   # Or using Node.js if you have it installed
   npx serve . -p 3000
   ```

## Usage

1. **Start both servers**:
   - Backend: `http://localhost:8000`
   - Frontend: `http://localhost:3000`

2. **Upload Study Materials**:
   - Navigate to the Upload section
   - Select a PDF file containing your study materials
   - Upload and wait for processing

3. **Ask Questions**:
   - Go to the Chat section
   - Select the subject and difficulty level
   - Type your question and get AI-powered answers

4. **Track Progress**:
   - Visit the Progress section to see your learning statistics
   - Get revision suggestions based on the forgetting curve

## API Endpoints

### Upload
- `POST /upload` - Upload and process PDF files

### Q&A
- `POST /ask` - Ask questions about uploaded materials
- `POST /explain` - Get feedback on your explanations

### Progress
- `GET /progress` - Get learning progress and forgetting curve status
- `GET /session-report` - Generate session summary

## Project Structure

```
SDP/
├── Backend/
│   ├── api/
│   │   └── routes.py          # API endpoints
│   ├── services/
│   │   ├── llm_service.py     # AI interactions
│   │   ├── pdf_service.py     # PDF processing
│   │   ├── progress_service.py # Progress tracking
│   │   ├── memory_service.py  # Session history
│   │   ├── semantic_service.py # Text similarity
│   │   ├── web_service.py     # Web fallback
│   │   └── ml/
│   │       ├── mastery_model.py    # Mastery prediction
│   │       └── forgetting_model.py # Forgetting curve
│   ├── utils/
│   │   └── text_utils.py      # Text processing utilities
│   ├── data/                  # Storage directory
│   ├── config.py              # Configuration
│   ├── main.py                # FastAPI application
│   └── requirements.txt       # Dependencies
└── Frontend/
    ├── index.html             # Main web interface
    └── script.js              # Frontend logic
```

## Configuration

The application can be configured through environment variables and the `config.py` file:

- `GEMINI_API_KEY`: Your Google Gemini API key (required)
- `CHUNK_SIZE`: Text chunk size for processing (default: 500)
- `MAX_CHUNKS_PER_QUERY`: Maximum chunks to consider per query (default: 3)

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your Gemini API key is valid and set in the `.env` file
2. **PDF Processing Error**: Ensure the PDF is text-based (not scanned images)
3. **CORS Issues**: The backend is configured to allow all origins for development
4. **Memory Issues**: For large PDFs, consider reducing the chunk size

### Getting Help

- Check the browser console for frontend errors
- Check the backend terminal for API errors
- Ensure all dependencies are properly installed
- Verify your API key has sufficient quota

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Future Enhancements

- [ ] Multi-language support
- [ ] More subject areas
- [ ] Advanced analytics dashboard
- [ ] Collaborative study features
- [ ] Mobile app
- [ ] Integration with learning management systems
