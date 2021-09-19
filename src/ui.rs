pub struct Settings {
    pub angle: f32,
    pub levels: u32,
}

pub fn update(ctx: &egui::CtxRef, setting: &mut Settings) {
    egui::SidePanel::right("Settings").show(ctx, |ui| {
        ui.add(egui::Label::new("Hello World!"));
        ui.label("A shorter and more convenient way to add a label.");
        if ui.button("Click me").clicked() { /* take some action here */ }
        ui.drag_angle(&mut setting.angle);
        ui.add(egui::DragValue::new(&mut setting.levels).clamp_range(0..=100))
    });
}
