import { create } from 'zustand';
import { v4 as uuidv4 } from 'uuid';

const useChatStore = create((set, get) => ({
    messages: [],
    isLoading: false,
    error: null,
    activeTopic: null, // For topic badges

    // Actions
    addMessage: (role, content, meta = {}) => {
        const newMessage = {
            id: uuidv4(),
            role,
            content,
            timestamp: new Date().toISOString(),
            ...meta
        };

        set((state) => ({
            messages: [...state.messages, newMessage],
            error: null // Clear previous errors on new message
        }));
    },

    updateLastMessage: (updates) => {
        set((state) => {
            const msgs = [...state.messages];
            if (msgs.length > 0) {
                const lastMsg = msgs[msgs.length - 1];
                msgs[msgs.length - 1] = { ...lastMsg, ...updates };
            }
            return { messages: msgs };
        });
    },

    setLoading: (loading) => set({ isLoading: loading }),

    setError: (error) => set({ error }),

    clearChat: () => set({ messages: [], error: null }),

    setActiveTopic: (topic) => set({ activeTopic: topic })
}));

export default useChatStore;
