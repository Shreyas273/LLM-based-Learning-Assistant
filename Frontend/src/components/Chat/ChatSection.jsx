import React, { useEffect, useRef } from 'react';
import ChatMessage from './ChatMessage';
import ChatInput from './MessageInput';
import useChatStore from '../../store/chatStore';
import { askAI, uploadFile } from '../../services/api';
const ChatSection = () => {
    const userId = "guest";
    const [studyMode, setStudyMode] = React.useState('default'); // default|story|flowchart|exam_prep
    // Zustand Store
    const {
        messages,
        addMessage,
        isLoading,
        setLoading,
        setError
    } = useChatStore();

    // Refs
    const scrollRef = useRef(null);
    const abortControllerRef = useRef(null);

    // Auto-scroll on new messages
    useEffect(() => {
        scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // Initial greeting if empty
    useEffect(() => {
        if (messages.length === 0) {
            addMessage('assistant', "Hello! I'm your AI Learning Assistant. How can I help you study today?");
        }
    }, [messages.length, addMessage]);

    const handleSend = async (text, file) => {
        if (!text && !file) return;

        // Cancel previous request if active
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }

        // Create new controller
        const controller = new AbortController();
        abortControllerRef.current = controller;

        // Optimistic UI Update
        const content = text || (file ? `Uploaded: ${file.name}` : '');
        addMessage('user', content, { file });
        setLoading(true);

        try {
            let response;
            let currentSubject = "general";

            if (file) {
                const loadingId = "loading-msg"; // We don't really use IDs for temp toast here but could
                // Just let the spinner verify loading state

                const formData = new FormData();
                formData.append('file', file);

                const uploadRes = await uploadFile(formData);
                currentSubject = uploadRes.file;

                addMessage('assistant', `✅ **File uploaded:** ${file.name}\n\nReady to analyze.`);

                if (text) {
                    response = await askAI({
                        question: text,
                        subject: currentSubject,
                        context_mode: 'pdf',
                        user_id: userId,
                        study_approach: studyMode
                    }, controller.signal);
                } else {
                    setLoading(false);
                    return;
                }

            } else {
                // Text only
                response = await askAI({
                    question: text,
                    subject: "general",
                    user_id: userId,
                    study_approach: studyMode
                }, controller.signal);
            }

            if (response) {
                addMessage('assistant', response.answer || "I processed that.", {
                    source_file: response.source_file,
                    confidence: response.confidence_score,
                    sources: response.sources,
                    mode: response.mode,
                    study_approach: studyMode
                });
            }

        } catch (error) {
            if (axios.isCancel(error)) {
                console.log('Request canceled', error.message);
            } else {
                console.error("Chat Error:", error);

                // Show error in chat
                let errorText = error.message || "An unexpected error occurred.";
                addMessage('assistant', `⚠️ **Error:** ${errorText}`);
                setError(errorText); // Set global error too
            }
        } finally {
            setLoading(false);
            abortControllerRef.current = null;
        }
    };

    return (
        <div className="flex flex-col h-[calc(100vh-8rem)]">
            {/* Messages Area */}
            <div className="flex-grow overflow-y-auto custom-scrollbar p-4 space-y-6">
                {messages.map((msg) => (
                    <ChatMessage key={msg.id} message={msg} />
                ))}

                {isLoading && (
                    <div className="flex justify-start animate-fade-in">
                        <div className="bg-slate-800/50 rounded-2xl p-4 flex items-center space-x-2">
                            <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                            <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                            <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                        </div>
                    </div>
                )}
                <div ref={scrollRef} />
            </div>

            {/* Input Area */}
            <div className="mt-4">
                <ChatInput
                    onSend={handleSend}
                    loading={isLoading}
                    studyMode={studyMode}
                    onStudyModeChange={setStudyMode}
                />
            </div>
        </div>
    );
};

// Need axios for isCancel check, or just check name
import axios from 'axios';

export default ChatSection;
