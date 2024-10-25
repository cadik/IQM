#ifndef DEBUG_UTILS_H
#define DEBUG_UTILS_H

#ifdef DEBUG
#include <renderdoc_app.h>
static bool renderdocActive = false;
#endif

inline void initRenderDoc() {
#ifdef DEBUG
    RENDERDOC_API_1_6_0 *rdoc_api = nullptr;
    if (void *mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD)) {
        std::cout << "Renderdoc loaded" << std::endl;
        pRENDERDOC_GetAPI RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)(dlsym(mod, "RENDERDOC_GetAPI"));
        int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_6_0, reinterpret_cast<void **>(&rdoc_api));
        assert(ret == 1);
        renderdocActive = true;
    }

    if (renderdocActive) {
        rdoc_api->StartFrameCapture(nullptr, nullptr);
    }
#endif
}

inline void finishRenderDoc() {
#ifdef DEBUG
    if (renderdocActive) {
        auto res = rdoc_api->EndFrameCapture(nullptr, nullptr);
        std::cout << (res ? "capture ok" : "capture not ok") << std::endl;
    }
#endif
}

#endif //DEBUG_UTILS_H
