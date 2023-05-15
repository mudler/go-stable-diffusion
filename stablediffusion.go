package stablediffusion

// #cgo CXXFLAGS: -I./ -I./ncnn/build/src/ -I./stable-diffusion/x86/linux/include/ -I./stable-diffusion/x86/linux/src/
// #cgo LDFLAGS: -L./ -lstablediffusion -lgomp -lopencv_core -lopencv_imgcodecs -lm -lstdc++
// #include "stablediffusion.h"
// #include <stdlib.h>
import "C"
import (
	"fmt"
)

func GenerateImage(height, width, mode, step, seed int, positive_prompt, negative_prompt, dst string) error {

	// int generate_image(int height, int width, int mode, int  step, int seed, const char *positive_prompt, const char *negative_prompt, const char *dst)

	pp := C.CString(positive_prompt)
	np := C.CString(negative_prompt)
	destination := C.CString(dst)

	ret := C.generate_image(C.int(height), C.int(width), C.int(mode), C.int(step), C.int(seed), pp, np, destination)
	if ret != 0 {
		return fmt.Errorf("failed")
	}
	return nil
}
