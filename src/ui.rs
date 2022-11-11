#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub n: u32,
    pub scale: f64,
    pub rotation: f32,
    pub busy_loop: bool,
}

pub fn update(ctx: &egui::Context, setting: &mut Settings, frame_time: f64) {
    egui::SidePanel::right("Settings").show(ctx, |ui| {
        ui.label("Rotation:");
        ui.drag_angle(&mut setting.rotation);
        ui.label("Polygon:");
        ui.add(
            egui::DragValue::new(&mut setting.n)
                .clamp_range(3..=10)
                .speed(0.01),
        );
        ui.label("Scale:");
        ui.add(
            egui::DragValue::new(&mut setting.scale)
                .clamp_range(-0.8..=0.8)
                .speed(0.0005),
        );
        ui.checkbox(&mut setting.busy_loop, "Busy Loop");
        if setting.busy_loop {
            ui.label(format!("FPS: {:.0}", 1.0 / frame_time));
            ui.label(format!("Frame Time: {:.3}ms", frame_time * 1000.0));
        }
    });
}
