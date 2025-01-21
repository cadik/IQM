/*
 * Image Quality Metrics
 * Petr Volf - 2024
 */

#ifndef DEBUG_UTILS_H
#define DEBUG_UTILS_H

#ifdef ENABLE_RENDERDOC
#include <renderdoc_app.h>
static bool renderdocActive = false;
static RENDERDOC_API_1_6_0 *api = nullptr;
#endif

inline void initRenderDoc() {
#ifdef ENABLE_RENDERDOC
    if (void *mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD)) {
        std::cout << "Renderdoc loaded" << std::endl;
        pRENDERDOC_GetAPI RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)(dlsym(mod, "RENDERDOC_GetAPI"));
        int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_6_0, reinterpret_cast<void **>(&api));
        assert(ret == 1);
        renderdocActive = true;
    }

    if (renderdocActive) {
        api->StartFrameCapture(nullptr, nullptr);
    }
#endif
}

inline void finishRenderDoc() {
#ifdef ENABLE_RENDERDOC
    if (renderdocActive) {
        auto res = api->EndFrameCapture(nullptr, nullptr);
        std::cout << (res ? "capture ok" : "capture not ok") << std::endl;
    }
#endif
}

#endif //DEBUG_UTILS_H
