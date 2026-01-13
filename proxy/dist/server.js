"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const node_http_1 = require("node:http");
const runtime_1 = require("@copilotkit/runtime");
const client_1 = require("@ag-ui/client");
const serviceAdapter = new runtime_1.OpenAIAdapter();
const server = (0, node_http_1.createServer)((req, res) => {
    const runtime = new runtime_1.CopilotRuntime({
        agents: {
            // Our FastAPI endpoint URL
            my_agent: new client_1.HttpAgent({ url: "http://localhost:8000/copilotkit" }),
        },
    });
    const handler = (0, runtime_1.copilotRuntimeNodeHttpEndpoint)({
        endpoint: "/copilotkit",
        runtime,
        serviceAdapter,
    });
    return handler(req, res);
});
server.listen(4000, () => {
    console.log("Listening at http://localhost:4000/copilotkit");
});
