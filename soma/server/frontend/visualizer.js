'use strict';

/**
 * Soma Space2D Visualizer
 * WebSocket-based visualization client for the Soma simulation environment.
 */

const CONFIG = {
    WS_URL: 'ws://localhost:8765',
    RECONNECT_INTERVAL: 3000,
    SCALE_FACTOR: 10,
    AGENT_COLOR: '#00d9ff',
    OBSTACLE_COLOR: '#ff6b6b',
    VELOCITY_COLOR: '#ffd93d',
    BOUNDARY_COLOR: '#3a3a5c',
    BACKGROUND_COLOR: '#16213e'
};

/**
 * Visualizer class handles WebSocket connection and canvas rendering.
 */
class Visualizer {
    /**
     * Creates a new Visualizer instance.
     */
    constructor() {
        this.canvas = document.getElementById('simulation-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.ws = null;
        this.state = {
            width: 100,
            height: 100,
            agents: [],
            obstacles: [],
            stepCount: 0,
            isRunning: false
        };
        this.isConnected = false;
        this.reconnectTimer = null;

        this.initCanvas();
        this.initControls();
        this.connect();
    }

    /**
     * Initializes the canvas with proper sizing.
     */
    initCanvas() {
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
    }

    /**
     * Resizes the canvas to fit its container.
     */
    resizeCanvas() {
        const container = this.canvas.parentElement;
        const rect = container.getBoundingClientRect();
        this.canvas.width = rect.width - 280;
        this.canvas.height = rect.height;

        if (window.innerWidth <= 768) {
            this.canvas.width = rect.width;
            this.canvas.height = rect.height - 200;
        }

        this.render();
    }

    /**
     * Initializes control button event listeners.
     */
    initControls() {
        document.getElementById('btn-start').addEventListener('click', () => this.start());
        document.getElementById('btn-stop').addEventListener('click', () => this.stop());
        document.getElementById('btn-reset').addEventListener('click', () => this.reset());
    }

    /**
     * Establishes WebSocket connection to the server.
     */
    connect() {
        this.updateConnectionStatus('connecting');

        try {
            this.ws = new WebSocket(CONFIG.WS_URL);

            this.ws.onopen = () => {
                this.isConnected = true;
                this.updateConnectionStatus('connected');
                console.log('Connected to server');
                if (this.reconnectTimer) {
                    clearTimeout(this.reconnectTimer);
                    this.reconnectTimer = null;
                }
            };

            this.ws.onclose = () => {
                this.isConnected = false;
                this.updateConnectionStatus('disconnected');
                console.log('Disconnected from server');
                this.scheduleReconnect();
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            this.ws.onmessage = (event) => {
                this.handleMessage(event.data);
            };
        } catch (error) {
            console.error('Failed to connect:', error);
            this.scheduleReconnect();
        }
    }

    /**
     * Schedules a reconnection attempt.
     */
    scheduleReconnect() {
        if (this.reconnectTimer) return;
        this.reconnectTimer = setTimeout(() => {
            this.reconnectTimer = null;
            this.connect();
        }, CONFIG.RECONNECT_INTERVAL);
    }

    /**
     * Updates the connection status indicator.
     * @param {string} status - The connection status ('connected', 'disconnected', 'connecting').
     */
    updateConnectionStatus(status) {
        const indicator = document.getElementById('status-indicator');
        const text = document.getElementById('status-text');

        indicator.className = status;

        const statusMessages = {
            connected: 'Connected',
            disconnected: 'Disconnected',
            connecting: 'Connecting...'
        };

        text.textContent = statusMessages[status] || 'Unknown';
    }

    /**
     * Handles incoming WebSocket messages.
     * @param {string} data - The raw message data.
     */
    handleMessage(data) {
        try {
            const message = JSON.parse(data);
            this.processMessage(message);
        } catch (error) {
            console.error('Failed to parse message:', error);
        }
    }

    /**
     * Processes a parsed WebSocket message.
     * @param {Object} message - The parsed message object.
     */
    processMessage(message) {
        switch (message.type) {
            case 'state':
                this.updateState(message.data);
                break;
            case 'init':
                this.initState(message.data);
                break;
            case 'step':
                this.state.stepCount = message.data.step_count;
                this.updateInfoPanel();
                break;
            default:
                console.log('Unknown message type:', message.type);
        }
    }

    /**
     * Initializes the simulation state.
     * @param {Object} data - The initialization data.
     */
    initState(data) {
        this.state.width = data.width || 100;
        this.state.height = data.height || 100;
        this.state.stepCount = data.step_count || 0;
        this.state.agents = this.parseAgents(data.agents || []);
        this.state.obstacles = this.parseObstacles(data.obstacles || []);
        this.updateInfoPanel();
        this.render();
    }

    /**
     * Updates the simulation state.
     * @param {Object} data - The state update data.
     */
    updateState(data) {
        if (data.width) this.state.width = data.width;
        if (data.height) this.state.height = data.height;
        if (data.step_count !== undefined) this.state.stepCount = data.step_count;
        if (data.agents) this.state.agents = this.parseAgents(data.agents);
        if (data.obstacles) this.state.obstacles = this.parseObstacles(data.obstacles);
        this.updateInfoPanel();
        this.render();
    }

    /**
     * Parses agent data from server format.
     * @param {Array} agents - Raw agent data array.
     * @returns {Array} Parsed agent objects.
     */
    parseAgents(agents) {
        return agents.map((agent) => ({
            uid: agent.uid,
            position: {
                x: agent.position[0],
                y: agent.position[1]
            },
            velocity: {
                x: agent.velocity[0],
                y: agent.velocity[1]
            },
            collisionRadius: agent.collision_radius,
            mass: agent.mass
        }));
    }

    /**
     * Parses obstacle data from server format.
     * @param {Array} obstacles - Raw obstacle data array.
     * @returns {Array} Parsed obstacle objects.
     */
    parseObstacles(obstacles) {
        return obstacles.map((obstacle) => ({
            position: {
                x: obstacle.position[0],
                y: obstacle.position[1]
            },
            collisionRadius: obstacle.collision_radius
        }));
    }

    /**
     * Updates the info panel with current state values.
     */
    updateInfoPanel() {
        document.getElementById('step-count').textContent = this.state.stepCount;
        document.getElementById('agent-count').textContent = this.state.agents.length;
        document.getElementById('obstacle-count').textContent = this.state.obstacles.length;
    }

    /**
     * Sends a start command to the server.
     */
    start() {
        if (!this.isConnected) return;
        this.state.isRunning = true;
        this.sendMessage({ type: 'command', action: 'start' });
    }

    /**
     * Sends a stop command to the server.
     */
    stop() {
        if (!this.isConnected) return;
        this.state.isRunning = false;
        this.sendMessage({ type: 'command', action: 'stop' });
    }

    /**
     * Sends a reset command to the server.
     */
    reset() {
        if (!this.isConnected) return;
        this.state.isRunning = false;
        this.sendMessage({ type: 'command', action: 'reset' });
    }

    /**
     * Sends a message to the server.
     * @param {Object} message - The message object to send.
     */
    sendMessage(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        }
    }

    /**
     * Converts world coordinates to canvas coordinates.
     * @param {number} x - World x coordinate.
     * @param {number} y - World y coordinate.
     * @returns {Object} Canvas coordinates {x, y}.
     */
    worldToCanvas(x, y) {
        const scaleX = this.canvas.width / this.state.width;
        const scaleY = this.canvas.height / this.state.height;
        const scale = Math.min(scaleX, scaleY) * 0.9;

        const offsetX = (this.canvas.width - this.state.width * scale) / 2;
        const offsetY = (this.canvas.height - this.state.height * scale) / 2;

        return {
            x: x * scale + offsetX,
            y: this.canvas.height - (y * scale + offsetY)
        };
    }

    /**
     * Converts a world scale value to canvas scale.
     * @param {number} value - The world scale value.
     * @returns {number} The canvas scale value.
     */
    scaleToCanvas(value) {
        const scaleX = this.canvas.width / this.state.width;
        const scaleY = this.canvas.height / this.state.height;
        const scale = Math.min(scaleX, scaleY) * 0.9;
        return value * scale;
    }

    /**
     * Renders the current state to the canvas.
     */
    render() {
        this.ctx.fillStyle = CONFIG.BACKGROUND_COLOR;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        this.drawBoundary();
        this.drawObstacles();
        this.drawAgents();
    }

    /**
     * Draws the environment boundary.
     */
    drawBoundary() {
        const topLeft = this.worldToCanvas(0, this.state.height);
        const bottomRight = this.worldToCanvas(this.state.width, 0);

        this.ctx.strokeStyle = CONFIG.BOUNDARY_COLOR;
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);
        this.ctx.strokeRect(
            topLeft.x,
            topLeft.y,
            bottomRight.x - topLeft.x,
            bottomRight.y - topLeft.y
        );
        this.ctx.setLineDash([]);
    }

    /**
     * Draws all obstacles on the canvas.
     */
    drawObstacles() {
        this.state.obstacles.forEach((obstacle) => {
            const pos = this.worldToCanvas(obstacle.position.x, obstacle.position.y);
            const radius = this.scaleToCanvas(obstacle.collisionRadius);

            this.ctx.beginPath();
            this.ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
            this.ctx.fillStyle = CONFIG.OBSTACLE_COLOR;
            this.ctx.fill();

            this.ctx.beginPath();
            this.ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
            this.ctx.strokeStyle = this.adjustColor(CONFIG.OBSTACLE_COLOR, -30);
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
        });
    }

    /**
     * Draws all agents on the canvas.
     */
    drawAgents() {
        this.state.agents.forEach((agent) => {
            const pos = this.worldToCanvas(agent.position.x, agent.position.y);
            const radius = this.scaleToCanvas(agent.collisionRadius);

            this.ctx.beginPath();
            this.ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
            this.ctx.fillStyle = CONFIG.AGENT_COLOR;
            this.ctx.fill();

            this.ctx.beginPath();
            this.ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
            this.ctx.strokeStyle = this.adjustColor(CONFIG.AGENT_COLOR, -30);
            this.ctx.lineWidth = 2;
            this.ctx.stroke();

            this.drawVelocityVector(agent, pos, radius);
        });
    }

    /**
     * Draws a velocity vector for an agent.
     * @param {Object} agent - The agent object.
     * @param {Object} pos - Canvas position of the agent.
     * @param {number} radius - Canvas radius of the agent.
     */
    drawVelocityVector(agent, pos, radius) {
        const velocityMagnitude = Math.sqrt(
            agent.velocity.x ** 2 + agent.velocity.y ** 2
        );

        if (velocityMagnitude < 0.01) return;

        const vectorLength = Math.min(velocityMagnitude * CONFIG.SCALE_FACTOR, radius * 3);
        const normalizedVx = agent.velocity.x / velocityMagnitude;
        const normalizedVy = agent.velocity.y / velocityMagnitude;

        const endX = pos.x + normalizedVx * vectorLength;
        const endY = pos.y - normalizedVy * vectorLength;

        this.ctx.beginPath();
        this.ctx.moveTo(pos.x, pos.y);
        this.ctx.lineTo(endX, endY);
        this.ctx.strokeStyle = CONFIG.VELOCITY_COLOR;
        this.ctx.lineWidth = 2;
        this.ctx.stroke();

        const arrowSize = 6;
        const angle = Math.atan2(-(normalizedVy), normalizedVx);

        this.ctx.beginPath();
        this.ctx.moveTo(endX, endY);
        this.ctx.lineTo(
            endX - arrowSize * Math.cos(angle - Math.PI / 6),
            endY - arrowSize * Math.sin(angle - Math.PI / 6)
        );
        this.ctx.lineTo(
            endX - arrowSize * Math.cos(angle + Math.PI / 6),
            endY - arrowSize * Math.sin(angle + Math.PI / 6)
        );
        this.ctx.closePath();
        this.ctx.fillStyle = CONFIG.VELOCITY_COLOR;
        this.ctx.fill();
    }

    /**
     * Adjusts a hex color by a brightness amount.
     * @param {string} color - The hex color string.
     * @param {number} amount - The brightness adjustment amount.
     * @returns {string} The adjusted hex color.
     */
    adjustColor(color, amount) {
        const hex = color.replace('#', '');
        const r = Math.max(0, Math.min(255, parseInt(hex.substr(0, 2), 16) + amount));
        const g = Math.max(0, Math.min(255, parseInt(hex.substr(2, 2), 16) + amount));
        const b = Math.max(0, Math.min(255, parseInt(hex.substr(4, 2), 16) + amount));
        return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.visualizer = new Visualizer();
});
