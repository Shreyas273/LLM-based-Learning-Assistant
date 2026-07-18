import { FileText, File, Image, PenTool, BookOpen, Brain, Calculator, FlaskConical, Atom, GraduationCap, Link, Hash } from 'lucide-react';

export const TOOLS = [
    // Summarizers
    { id: 'pdf-summarizer', name: 'PDF Summarizer', icon: FileText, color: 'text-red-400', type: 'summarizer', subType: 'pdf', input: 'file', accept: '.pdf', description: 'Summarize PDF documents efficiently.' },
    { id: 'word-summarizer', name: 'Word Summarizer', icon: File, color: 'text-blue-400', type: 'summarizer', subType: 'docx', input: 'file', accept: '.docx', description: 'Extract insights from Word docs.' },
    { id: 'ppt-summarizer', name: 'PPT Summarizer', icon: FileText, color: 'text-orange-400', type: 'summarizer', subType: 'pptx', input: 'file', accept: '.pptx', description: 'Get key takeaways from slides.' },
    { id: 'text-summarizer', name: 'Text Summarizer', icon: FileText, color: 'text-gray-400', type: 'summarizer', subType: 'txt', input: 'file', accept: '.txt', description: 'Summarize plain text files.' },
    { id: 'image-summarizer', name: 'Image Summarizer', icon: Image, color: 'text-purple-400', type: 'summarizer', subType: 'image', input: 'file', accept: 'image/*', description: 'Analyze and describe images.' },

    // Generators
    { id: 'humanizer', name: 'AI Humanizer', icon: PenTool, color: 'text-green-400', type: 'generator', subType: 'humanizer', input: 'text', placeholder: 'Paste text to humanize...', description: 'Make AI text sound natural.' },
    { id: 'paper-writer', name: 'AI Paper Writer', icon: BookOpen, color: 'text-indigo-400', type: 'generator', subType: 'paper_writer', input: 'text', placeholder: 'Enter topic for paper...', description: 'Generate academic papers.' },
    { id: 'essay-writer', name: 'AI Essay Writer', icon: PenTool, color: 'text-pink-400', type: 'generator', subType: 'essay_writer', input: 'text', placeholder: 'Enter essay topic...', description: 'Write structured essays.' },
    { id: 'mind-map', name: 'Mind Map Generator', icon: Brain, color: 'text-yellow-400', type: 'generator', subType: 'mind_map', input: 'text', placeholder: 'Enter central topic...', description: 'Visualize ideas hierarchically.' },
    { id: 'quiz-generator', name: 'AI Quiz Generator', icon: Hash, color: 'text-cyan-400', type: 'generator', subType: 'quiz', input: 'text', placeholder: 'Enter subject for quiz...', description: 'Create multiple-choice quizzes.' },
    { id: 'answer-generator', name: 'AI Answer Generator', icon: Link, color: 'text-teal-400', type: 'generator', subType: 'answer', input: 'text', placeholder: 'Ask a question...', description: 'Get direct answers instantly.' },

    // Solvers
    { id: 'math-solver', name: 'AI Math Solver', icon: Calculator, color: 'text-blue-500', type: 'solver', subType: 'math', input: 'mixed', placeholder: 'Enter math problem...', description: 'Step-by-step math solutions.' },
    { id: 'chemistry-solver', name: 'Chemistry Solver', icon: FlaskConical, color: 'text-green-500', type: 'solver', subType: 'chemistry', input: 'mixed', placeholder: 'Enter reaction or problem...', description: 'Balance equations & solve problems.' },
    { id: 'physics-solver', name: 'Physics Solver', icon: Atom, color: 'text-purple-500', type: 'solver', subType: 'physics', input: 'mixed', placeholder: 'Enter physics problem...', description: 'Explain physics concepts.' },
    { id: 'homework-helper', name: 'Homework Helper', icon: GraduationCap, color: 'text-orange-500', type: 'solver', subType: 'homework', input: 'mixed', placeholder: 'Describe homework task...', description: 'Your personal AI tutor.' },
    { id: 'graphing-calc', name: 'Graphing Calculator', icon: Calculator, color: 'text-red-500', type: 'solver', subType: 'graphing', input: 'text', placeholder: 'Enter function to graph...', description: 'Analyze functions for graphing.' },
];
