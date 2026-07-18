import React, { useState, Suspense, lazy } from 'react';
import Header from './components/Layout/Header';
import ChatSection from './components/Chat/ChatSection';
import ToolsSection from './components/Tools/ToolsSection';
import { motion, AnimatePresence } from 'framer-motion';

// Lazy Load Heavy Dashboard Components
const AnalyticsSection = lazy(() => import('./components/Dashboard/AnalyticsSection'));
const MasterySection = lazy(() => import('./components/Dashboard/MasterySection'));
const KnowledgeSection = lazy(() => import('./components/Dashboard/KnowledgeSection'));
const ExplainBackToMeSection = lazy(() => import('./components/Explain/ExplainBackToMeSection'));

function App() {
    const [activeTab, setActiveTab] = useState('chat');

    const renderContent = () => {
        // Shared fallback loader
        const fallback = (
            <div className="flex h-64 items-center justify-center">
                <div className="w-8 h-8 border-4 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin" />
            </div>
        );
        switch (activeTab) {
            case 'chat': return <ChatSection />;
            case 'explain': return <Suspense fallback={fallback}><ExplainBackToMeSection /></Suspense>;
            case 'tools': return <ToolsSection />;
            case 'analytics':
                return <Suspense fallback={fallback}><AnalyticsSection /></Suspense>;
            case 'mastery':
                return <Suspense fallback={fallback}><MasterySection /></Suspense>;
            case 'knowledge':
                return <Suspense fallback={fallback}><KnowledgeSection /></Suspense>;
            default: return <ChatSection />;
        }
    };

    return (
        <div className="min-h-screen text-slate-50 selection:bg-sky-500/30 font-sans">
            <Header
                activeTab={activeTab}
                setActiveTab={setActiveTab}
            />
            <main className="max-w-7xl mx-auto px-4 py-6">
                <AnimatePresence mode="wait">
                    <motion.div
                        key={activeTab}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        transition={{ duration: 0.3, ease: "easeInOut" }}
                        className="w-full"
                    >
                        {renderContent()}
                    </motion.div>
                </AnimatePresence>
            </main>
        </div>
    );
}

export default App;
