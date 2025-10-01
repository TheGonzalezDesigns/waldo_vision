pub mod image_converter {
    use crate::core_modules::D1::pixel::pixel::*;

    pub fn convert_image(img: image::DynamicImage) {
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();
        let frame: Vec<u8> = rgba.into_raw();
        assert_eq!(frame.len(), (w * h * 4) as usize);
        let pixels: Pixels = Pixels::try_from(frame).expect("Failed to convert: ");
    }
}
