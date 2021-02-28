use winit::dpi::PhysicalSize;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Accumulate {
    pub instance_levels: u32,
    pub mesh_levels: u32,
    pub size: PhysicalSize<u32>,
    pub name: String,
}

/// Scene independent plan. Redone on window resize.
#[derive(Debug)]
pub struct Plan {
    pub passes: Vec<Accumulate>,
}
