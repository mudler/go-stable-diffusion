#ifdef __cplusplus
#include <vector>
#include <string>
extern "C" {
#endif

#include <stdbool.h>

int generate_image(int height, int width, int mode, int  step, int seed, const char *positive_prompt, const char *negative_prompt, const char *dst, const char *init_image, const char *assets_dir);
int generate_image_upscaled( int height, int width, int  step, int seed, const char *positive_prompt, const char *negative_prompt, const char *dst, const char *assets_dir);

#ifdef __cplusplus
}
#endif
