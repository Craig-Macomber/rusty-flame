//! Misc wrapper types for providing Clone, Debug and Eq

use std::{hash::Hash, ops::Deref, rc::Rc, sync::Arc};

#[derive(Debug, Clone)]
/// Wrapper that provides Debug.
pub struct DebugIt<T>(pub T);

impl<T> Deref for DebugIt<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug)]
/// Rc with pointer semantics (reference equality)
pub struct PtrArc<T>(Arc<T>);

// TODO: why does derive clone not work for this?
impl<T> Clone for PtrArc<T> {
    fn clone(&self) -> Self {
        PtrArc(self.0.clone())
    }
}

impl<T> PartialEq for PtrArc<T> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl<T> Deref for PtrArc<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> Hash for PtrArc<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let ptr = &*self.0 as *const T;
        ptr.hash(state);
    }
}

impl<T> Eq for PtrArc<T> {}

impl<T> From<T> for PtrArc<T> {
    fn from(t: T) -> Self {
        PtrArc(Arc::new(t))
    }
}
