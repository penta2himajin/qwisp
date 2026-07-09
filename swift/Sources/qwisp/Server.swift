import Foundation
import Hummingbird

// OpenAI-compatible HTTP server (productization step 5c) over the token-id backend.
// Increment 1: transport skeleton + /v1/models. /v1/chat/completions (SSE) follows.

struct ModelObject: ResponseEncodable {
    let id: String
    let object = "model"
    let created = 0
    let owned_by = "qwisp"
}

struct ModelsResponse: ResponseEncodable {
    let object = "list"
    let data: [ModelObject]
}

func makeRouter(modelID: String) -> Router<BasicRequestContext> {
    let router = Router()
    router.get("/v1/models") { _, _ in
        ModelsResponse(data: [ModelObject(id: modelID)])
    }
    return router
}

func runServe(modelID: String, port: Int) async throws {
    let app = Application(
        router: makeRouter(modelID: modelID),
        configuration: .init(address: .hostname("127.0.0.1", port: port))
    )
    print("qwisp serve → http://127.0.0.1:\(port)  (model: \(modelID))")
    try await app.runService()
}
