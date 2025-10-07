#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub x_coordinate: u32,
    pub y_coordinate: u32,
    pub width: u32,
    pub height: u32,
}

impl Region {
    pub fn new(x_coordinate: u32, y_coordinate: u32, width: u32, height: u32) -> Self {
        Self {
            x_coordinate,
            y_coordinate,
            width,
            height,
        }
    }

    pub fn center_coordinates(&self, frame_width: u32, frame_height: u32) -> (u32, u32) {
        let center_x = (self.x_coordinate + self.width / 2).min(frame_width.saturating_sub(1));
        let center_y = (self.y_coordinate + self.height / 2).min(frame_height.saturating_sub(1));
        (center_x, center_y)
    }
}
