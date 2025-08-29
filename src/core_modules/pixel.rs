mod pixel {
    type Byte = u8;
    type Bytes = Vec<Byte>;
    type Channel = Byte;
    type Luminance = f64;

    const CHANNELS: usize = 4;

    struct Pixel {
        pub red: Channel,
        pub green: Channel,
        pub blue: Channel,
        pub alpha: Channel,
    }

    impl Default for Pixel {
        fn default() -> Self {
            Pixel {
                red: Channel::default(),
                green: Channel::default(),
                blue: Channel::default(),
                alpha: Channel::default(),
            }
        }
    }

    impl Pixel {
        pub fn new(red: Channel, green: Channel, blue: Channel, alpha: Channel) -> Self {
            Pixel {
                red,
                green,
                blue,
                alpha,
            }
        }

        pub fn luminance(&self) -> Luminance {
            0.299 * self.red as f64 + 0.587 * self.green as f64 + 0.114 * self.blue as f64
        }
    }

    impl From<&[Byte]> for Pixel {
        fn from(bytes: &[Byte]) -> Self {
            if bytes.len() != CHANNELS {
                panic!("Cannot convert {} bytes into pixel.", bytes.len());
            }
            Pixel::new(bytes[0], bytes[1], bytes[2], bytes[3])
        }
    }

    impl From<Pixel> for Bytes {
        fn from(pixel: Pixel) -> Self {
            vec![pixel.red, pixel.green, pixel.blue, pixel.alpha]
        }
    }
}
