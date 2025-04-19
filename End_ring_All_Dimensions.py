import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# Checking the observed dimensions(just the outer-diameter in this case) of the component against the specification values.
def check_tolerance(measured, nominal, tolerance):
    return abs(measured - nominal) <= tolerance


#Preprocessing, Noice reduction and Acquiring dimensions 
def measure_end_ring(image_path, calibration_factor=0.1):
    # Load the image
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found.")

    # Sort contours by area (largest is outer ring, next is inner ring)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    outer_contour = contours[0]
    inner_contour = contours[1] if len(contours) > 1 else None

    # Outer diameter
    (x_out, y_out), outer_radius = cv2.minEnclosingCircle(outer_contour)
    outer_diameter = 2 * outer_radius * calibration_factor

    # Inner diameter
    if inner_contour is not None:
        (x_in, y_in), inner_radius = cv2.minEnclosingCircle(inner_contour)
        inner_diameter = 2 * inner_radius * calibration_factor
    else:
        inner_diameter = None

    # Teeth detection using simplified contour to avoid self-intersections
    epsilon = 0.01 * cv2.arcLength(outer_contour, True)
    approx_contour = cv2.approxPolyDP(outer_contour, epsilon, True)

    teeth_thicknesses = []
    try:
        hull = cv2.convexHull(approx_contour, returnPoints=False)
        if hull is not None and len(hull) > 3:
            defects = cv2.convexityDefects(approx_contour, hull)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(approx_contour[s][0])
                    end = tuple(approx_contour[e][0])
                    dist = np.linalg.norm(np.array(start) - np.array(end))
                    teeth_thicknesses.append(dist * calibration_factor)
    except cv2.error as e:
        print("Convexity defect analysis failed:", e)
        teeth_thicknesses = []

    avg_teeth_thickness = np.mean(teeth_thicknesses) if teeth_thicknesses else None

    # Draw result
    output_image = image.copy()
    cv2.circle(output_image, (int(x_out), int(y_out)), int(outer_radius), (0, 255, 0), 2)
    if inner_contour is not None:
        cv2.circle(output_image, (int(x_in), int(y_in)), int(inner_radius), (255, 0, 0), 2)

    # Display result
    cv2.imshow("Detected End Ring Dimensions", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Printing the observed values of the component 
    df = pd.DataFrame({
        "Measurement": ["Outer Diameter", "Inner Diameter", "Avg Teeth Thickness"],
        "Value (mm)": [outer_diameter, inner_diameter, avg_teeth_thickness]
    })
    print("\nMeasured Dimensions (in mm):")
    print(df)
    if check_tolerance(outer_diameter, 120.0, 0.2):
        print("Outer Diameter: PASS")
    else:
        print("Outer Diameter: FAIL")

    return outer_diameter, inner_diameter, avg_teeth_thickness


# Run the function
if __name__ == "__main__":
    test_image_path = "/home/guhan/Downloads/endring_image.jpeg" #path to the test image 
    measure_end_ring(test_image_path, calibration_factor=0.1)

