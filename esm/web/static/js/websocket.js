/**
 * WebSocketClient - Handle generation streaming
 */
class WebSocketClient {
    constructor() {
        this.ws = null;
        this.sessionId = null;
        this.callbacks = {
            start: null,
            phase: null,
            progress: null,
            sequence_complete: null,
            conformation_ready: null,
            function_complete: null,
            complete: null,
            error: null,
        };
    }

    /**
     * Connect to generation WebSocket
     */
    connect(sessionId) {
        this.sessionId = sessionId;

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${protocol}//${window.location.host}/ws/generate/${sessionId}`;

        this.ws = new WebSocket(url);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this._handleMessage(data);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            if (this.callbacks.error) {
                this.callbacks.error({ message: 'WebSocket connection error' });
            }
        };

        this.ws.onclose = () => {
            console.log('WebSocket closed');
        };
    }

    /**
     * Handle incoming message
     */
    _handleMessage(data) {
        switch (data.type) {
            case 'start':
                if (this.callbacks.start) this.callbacks.start(data);
                break;
            case 'phase':
                if (this.callbacks.phase) this.callbacks.phase(data);
                break;
            case 'progress':
                if (this.callbacks.progress) this.callbacks.progress(data);
                break;
            case 'sequence_complete':
                if (this.callbacks.sequence_complete) this.callbacks.sequence_complete(data);
                break;
            case 'conformation_ready':
                if (this.callbacks.conformation_ready) this.callbacks.conformation_ready(data);
                break;
            case 'function_complete':
                if (this.callbacks.function_complete) this.callbacks.function_complete(data);
                break;
            case 'complete':
                if (this.callbacks.complete) this.callbacks.complete(data);
                break;
            case 'error':
                if (this.callbacks.error) this.callbacks.error(data);
                break;
            default:
                console.warn('Unknown message type:', data.type);
        }
    }

    /**
     * Set callback handlers
     */
    on(event, callback) {
        if (event in this.callbacks) {
            this.callbacks[event] = callback;
        }
        return this;  // Allow chaining
    }

    /**
     * Close connection
     */
    close() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}

// Global instance
window.wsClient = new WebSocketClient();
