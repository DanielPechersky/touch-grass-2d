from dataclasses import dataclass

import numpy as np
from OpenGL import GL as gl
from PIL import Image


@dataclass
class GlTexture:
    id: int
    w: int
    h: int

    @property
    def dims(self):
        return np.array([self.w, self.h], dtype=np.float32)

    @staticmethod
    def load_texture_rgba(img: Image.Image):
        # Load with Pillow and upload to GL texture. Must be called AFTER GL context is created.
        rgba = np.array(img, dtype=np.uint8)  # H x W x 4
        h, w = rgba.shape[:2]

        tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        # Avoid row alignment issues for widths not multiple of 4
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA,
            w,
            h,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            rgba,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        return GlTexture(tex, w, h)
