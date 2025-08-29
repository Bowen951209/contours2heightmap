use ab_glyph::FontVec;
use log::debug;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub fn load_sans() -> FontVec {
    let font_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/OpenSans-Medium.ttf");

    debug!("Loading font from: {:?}", font_path);

    let mut data = Vec::new();
    File::open(&font_path)
        .expect("Failed to open font file")
        .read_to_end(&mut data)
        .expect("Failed to read font file");

    FontVec::try_from_vec(data).expect("Failed to load font")
}
