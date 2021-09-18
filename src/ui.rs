pub fn update(ctx: &egui::CtxRef) {
    egui::SidePanel::right("Settings").show(ctx, |ui| {
        ui.add(egui::Label::new("Hello World!"));
        ui.label("A shorter and more convenient way to add a label.");
        if ui.button("Click me").clicked() { /* take some action here */ }
    });
}
