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

const SMALL_ACCUMULATION_BUFFER_SIZE: u32 = 256;
const MID_ACCUMULATION_BUFFER_SIZE: u32 = 512;
const LARGE_ACCUMULATION_BUFFER_SIZE: u32 = 1024;
const LARGE_ACCUMULATION_BUFFER_SIZE2: u32 = 1800;

pub fn plan_render(size: winit::dpi::PhysicalSize<u32>) -> Plan {
    Plan {
        passes: vec![
            Accumulate {
                instance_levels: 4,
                mesh_levels: 4,
                size: winit::dpi::PhysicalSize {
                    width: 64,
                    height: 64,
                },
                name: "XXSmall".to_owned(),
            },
            Accumulate {
                instance_levels: 4,
                mesh_levels: 4,
                size: winit::dpi::PhysicalSize {
                    width: 128,
                    height: 128,
                },
                name: "XSmall".to_owned(),
            },
            Accumulate {
                instance_levels: 2,
                mesh_levels: 2,
                size: winit::dpi::PhysicalSize {
                    width: SMALL_ACCUMULATION_BUFFER_SIZE,
                    height: SMALL_ACCUMULATION_BUFFER_SIZE,
                },
                name: "Small".to_owned(),
            },
            Accumulate {
                instance_levels: 2,
                mesh_levels: 2,
                size: winit::dpi::PhysicalSize {
                    width: MID_ACCUMULATION_BUFFER_SIZE,
                    height: MID_ACCUMULATION_BUFFER_SIZE,
                },
                name: "Mid".to_owned(),
            },
            Accumulate {
                instance_levels: 2,
                mesh_levels: 2,
                size: winit::dpi::PhysicalSize {
                    width: LARGE_ACCUMULATION_BUFFER_SIZE,
                    height: LARGE_ACCUMULATION_BUFFER_SIZE,
                },
                name: "Large".to_owned(),
            },
            Accumulate {
                instance_levels: 2,
                mesh_levels: 1,
                size: winit::dpi::PhysicalSize {
                    width: LARGE_ACCUMULATION_BUFFER_SIZE2,
                    height: LARGE_ACCUMULATION_BUFFER_SIZE2,
                },
                name: "Large2".to_owned(),
            },
            Accumulate {
                instance_levels: 2,
                mesh_levels: 1,
                size,
                name: "Main".to_owned(),
            },
        ],
    }
}
