import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE,
    headers: {
        'Content-Type': 'application/json',
    },
    timeout: 60000, // 60s timeout
});

// Request Interceptor (Optional: Add Auth tokens here)
api.interceptors.request.use((config) => {
    return config;
}, (error) => Promise.reject(error));

// Response Interceptor (Global Error Handling)
api.interceptors.response.use(
    (response) => response.data, // Return data directly
    (error) => {
        // Standardize Error Object
        const customError = {
            message: error.response?.data?.detail || "Network Error. Please check your connection.",
            status: error.response?.status || 500,
            original: error
        };
        return Promise.reject(customError);
    }
);

// Endpoints
export const askAI = async (data, signal) => {
    return api.post('/ask', data, { signal });
};

export const explainConcept = async (data) => {
    return api.post('/explain', data);
};

export const getKnowledgeGraph = async () => {
    return api.get('/knowledge-graph');
};

export const uploadFile = async (formData) => {
    return api.post('/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
    });
};

// Tools
export const summarizeFile = async (formData) => {
    return api.post('/tools/upload-summarize', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
    });
};

export const generateContent = async (data) => {
    return api.post('/tools/generate', data);
};

export const solveProblem = async (data) => {
    return api.post('/tools/solve', data);
};

export const solveImageProblem = async (formData) => {
    return api.post('/tools/image-solve', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
    });
};

// Analytics
export const getProgress = async (userId) => {
    const id = userId || 'guest';
    return api.get(`/progress/${encodeURIComponent(id)}`);
};

export const getSessionReport = async (userId) => {
    const id = userId || 'guest';
    return api.get(`/session-report/${encodeURIComponent(id)}`);
};

export const getPdfIndex = async () => {
    return api.get('/pdf-index');
};

export const comparativeAnalysis = async (data) => {
    return api.post('/comparative-analysis', data);
};

export default api;
