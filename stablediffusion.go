package stablediffusion

// #cgo CXXFLAGS: -I./ -I./ncnn/src -I./ncnn -I./ncnn/build/src/ -I./stable-diffusion/x86/vs2019_opencv-mobile_ncnn-dll_demo/vs2019_opencv-mobile_ncnn-dll_demo/ -std=c++17
// #cgo LDFLAGS: -L./ -lstablediffusion -lgomp -lopencv_core -lopencv_imgcodecs -lm -lstdc++
// #include "stablediffusion.h"
// #include <stdlib.h>
import "C"
import (
	"fmt"
)

func GenerateImage(height, width, mode, step, seed int, positive_prompt, negative_prompt, dst, init_image, asset_dir string) error {
	pp := C.CString(positive_prompt)
	np := C.CString(negative_prompt)
	ii := C.CString(init_image)
	ad := C.CString(asset_dir)

	destination := C.CString(dst)

	ret := C.generate_image(C.int(height), C.int(width), C.int(mode), C.int(step), C.int(seed), pp, np, destination, ii, ad)
	if ret != 0 {
		return fmt.Errorf("failed")
	}
	return nil
}

func GenerateImageUpscaled(height, width, step, seed int, positive_prompt, negative_prompt, dst, asset_dir string) error {
	pp := C.CString(positive_prompt)
	np := C.CString(negative_prompt)
	ad := C.CString(asset_dir)

	destination := C.CString(dst)

	ret := C.generate_image_upscaled(C.int(height), C.int(width), C.int(step), C.int(seed), pp, np, destination, ad)
	if ret != 0 {
		return fmt.Errorf("failed")
	}
	return nil
}
