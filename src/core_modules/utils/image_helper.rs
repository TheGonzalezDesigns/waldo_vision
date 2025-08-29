mod image_helper {
    use image::ImageEncoder;
    use std::io::Write;

    pub fn save(
        name: String,
        width: u32,
        height: u32,
        buffer: &Vec<u8>,
    ) -> Result<(), image::error::ImageError> {
        let output = std::fs::File::create(name)?;
        let encoder = image::codecs::png::PngEncoder::new(output);

        encoder.write_image(buffer, height, width, image::ExtendedColorType::Rgba8)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use super::image_helper::*;

    #[test]
    fn save_white_file() {
        let height = 500u32;
        let width = 500u32;
        let buffer_size = (width * height * 4) as usize;
        let buffer = vec![255u8; buffer_size];
        let name = String::from("white_file.png");

        save(name, width, height, &buffer).expect("Error Saving File.");
    }

    #[test]
    fn save_gradient_file() {
        let height = 500u32;
        let width = 500u32;
        let buffer_size = (width * height * 4) as usize;
        let mut buffer = vec![255u8; buffer_size];
        let name = String::from("gradient_file.png");
        let mut intensity = 0;

        for i in buffer.chunks_mut(4) {
            i[0] = intensity;
            i[1] = intensity;
            i[2] = intensity;
            intensity += 1;
            intensity %= 255;
        }

        save(name, width, height, &buffer).expect("Error Saving File.");
    }

    #[test]
    fn save_chunked_gradient_file() {
        let height = 5000u32;
        let width = 5000u32;
        let channels = 4;
        let buffer_size = (width * height * channels) as usize;
        let mut buffer = vec![255u8; buffer_size];
        let name = String::from("chunky_gradient_file.png");

        let chunk_height = 500u32;
        let chunk_width = 500u32;

        let get_chunk_pixels = |start_byte_pos: usize| -> Vec<usize> {
            let mut v = Vec::<usize>::new();
            for i in 0..(chunk_width * chunk_height) {
                let row_in_chunk = i / chunk_width;
                let col_in_chunk = i % chunk_width;
                let pixel_offset = (width * row_in_chunk) + col_in_chunk;
                v.push(start_byte_pos + (pixel_offset * channels) as usize);
            }
            v
        };

        let max_chunks_x = width / chunk_width;
        let max_chunks_y = height / chunk_height;
        let max_chunks = max_chunks_x * max_chunks_y;

        let mut gray_scale_intensity = 0;

        for chunk_index in 0..max_chunks {
            let chunk_x = chunk_index % max_chunks_x;
            let chunk_y = chunk_index / max_chunks_x;

            let top_left_pixel_offset = (chunk_x * chunk_width) + (chunk_y * chunk_height * width);
            let start_byte_pos = top_left_pixel_offset * channels;

            let chunk_pixels = get_chunk_pixels(start_byte_pos as usize);
            for pixel_byte_start in chunk_pixels {
                buffer[pixel_byte_start] -= gray_scale_intensity;
                buffer[pixel_byte_start + 1] -= gray_scale_intensity;
                buffer[pixel_byte_start + 2] -= gray_scale_intensity;
            }
            gray_scale_intensity = (gray_scale_intensity + 1) % 255;
        }

        save(name, width, height, &buffer).expect("Error Saving File.");
    }
}
