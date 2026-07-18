import React from 'react';
import { Brain, MessageSquare, BarChart2, Trophy, Network, Wrench, MessageCircle } from 'lucide-react';

const Header = ({ activeTab, setActiveTab }) => {
    const tabs = [
        { id: 'chat', label: 'Chat', icon: MessageSquare, color: 'text-sky-400' },
        { id: 'explain', label: 'Explain Back to Me', icon: MessageCircle, color: 'text-orange-400' },
        { id: 'analytics', label: 'Analytics', icon: BarChart2, color: 'text-emerald-400' },
        { id: 'mastery', label: 'Mastery', icon: Trophy, color: 'text-amber-400' },
        { id: 'knowledge', label: 'Knowledge', icon: Network, color: 'text-violet-400' },
        { id: 'tools', label: 'AI Tools', icon: Wrench, color: 'text-pink-400' },
    ];

    return (
        <header className="sticky top-0 z-50 backdrop-blur-xl bg-slate-900/70 border-b border-slate-800/70">
            <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
                <div className="flex items-center space-x-3">
                    <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-sky-500 via-indigo-500 to-fuchsia-500 flex items-center justify-center shadow-lg shadow-sky-500/20">
                        <Brain className="h-5 w-5 text-white" />
                    </div>
                    <div>
                        <div className="flex items-center space-x-2">
                            <h1 className="text-lg font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
                                AI Learning Studio
                            </h1>
                            <span className="px-2 py-0.5 rounded-full bg-sky-500/10 border border-sky-500/20 text-[10px] font-medium text-sky-300 uppercase tracking-wider">
                                Titan
                            </span>
                        </div>
                    </div>
                </div>

                <nav className="hidden md:flex items-center space-x-1">
                    {tabs.map((tab) => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`nav-btn flex items-center space-x-2 
                                ${activeTab === tab.id ? 'active' : 'text-slate-400 hover:text-slate-200'}`}
                        >
                            <tab.icon className={`h-4 w-4 ${activeTab === tab.id ? 'text-white' : tab.color}`} />
                            <span>{tab.label}</span>
                        </button>
                    ))}
                </nav>
            </div>
        </header>
    );
};

export default Header;
