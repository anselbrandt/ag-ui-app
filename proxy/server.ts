import { createServer } from "node:http";
import {
  CopilotRuntime,
  OpenAIAdapter,
  copilotRuntimeNodeHttpEndpoint,
} from "@copilotkit/runtime";
import { HttpAgent } from "@ag-ui/client";

const serviceAdapter = new OpenAIAdapter();

const server = createServer((req, res) => {
  const runtime = new CopilotRuntime({
    agents: {
      // Our FastAPI endpoint URL
      my_agent: new HttpAgent({ url: "http://localhost:8000/copilotkit" }),
    },
  });
  const handler = copilotRuntimeNodeHttpEndpoint({
    endpoint: "/copilotkit",
    runtime,
    serviceAdapter,
  });

  return handler(req, res);
});

server.listen(4000, () => {
  console.log("Listening at http://localhost:4000/copilotkit");
});
