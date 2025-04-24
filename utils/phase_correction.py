import numpy as np

def phase_correction(complex_image, method='oval', roi='mask', phase_calc='middle'):
    """
    Perform phase correction on a complex MRI image with various options.
    
    Parameters:
    - complex_image: 3D complex numpy array (x, y, z)
    - method: 'threshold' for intensity-based masking, 'oval' for predefined oval mask
    - roi: 'full' to use the entire slice, 'mask' to use only the masked region
    - phase_calc: 'middle' to use only the middle slice for phase calculation,
                  'slice_by_slice' to calculate phase for each slice separately
    
    Returns:
    - corrected_image: Phase-corrected complex image
    """
    
    def create_threshold_mask(slice_2d):
        magnitude = np.abs(slice_2d)
        threshold = np.mean(magnitude) + 0.5 * np.std(magnitude)
        return magnitude > threshold
    
    def create_oval_mask(shape, center, radii):
        y, x = np.ogrid[:shape[0], :shape[1]]
        return ((x - center[0])**2 / radii[0]**2 + 
                (y - center[1])**2 / radii[1]**2 <= 1)
    
    def calculate_mean_phase(slice_2d, mask):
        if roi == 'full':
            return np.angle(slice_2d).mean()
        elif roi == 'mask':
            return np.angle(slice_2d[mask]).mean()
    
    corrected_image = np.zeros_like(complex_image)
    
    if phase_calc == 'middle':
        middle_slice = complex_image[:, :, complex_image.shape[2]//2]
        
        if method == 'threshold':
            mask = create_threshold_mask(middle_slice)
        elif method == 'oval':
            center = (middle_slice.shape[0]//2, middle_slice.shape[1]//2)
            radii = (middle_slice.shape[0]//4, middle_slice.shape[1]//4)
            mask = create_oval_mask(middle_slice.shape, center, radii)
        
        mean_phase = calculate_mean_phase(middle_slice, mask)
        corrected_image = complex_image * np.exp(-1j * mean_phase)
    
    elif phase_calc == 'slice_by_slice':
        for z in range(complex_image.shape[2]):
            slice_2d = complex_image[:, :, z]
            
            if method == 'threshold':
                mask = create_threshold_mask(slice_2d)
            elif method == 'oval':
                center = (slice_2d.shape[0]//2, slice_2d.shape[1]//2)
                radii = (slice_2d.shape[0]//4, slice_2d.shape[1]//4)
                mask = create_oval_mask(slice_2d.shape, center, radii)
            
            mean_phase = calculate_mean_phase(slice_2d, mask)
            corrected_image[:, :, z] = slice_2d * np.exp(-1j * mean_phase)
    
    return corrected_image

def create_circular_mask(complex_image, radius_factor=0.25):
    """
    Create a circular mask on the middle slice based on image dimensions.
    
    Parameters:
    - complex_image: 3D complex numpy array (x, y, z)
    - radius_factor: Factor to determine radius as a fraction of min dimension
    
    Returns:
    - mask: 2D boolean array
    """
    middle_slice = complex_image[:, :, complex_image.shape[2]//2]
    
    y, x = np.ogrid[:middle_slice.shape[0], :middle_slice.shape[1]]
    center = (middle_slice.shape[0]//2, middle_slice.shape[1]//2)
    radius = min(middle_slice.shape[0], middle_slice.shape[1]) * radius_factor
    
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    
    return mask 