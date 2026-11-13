def calculate_tumor_area(width, height):
    return width * height


def calculate_tumor_percentage(tumor_area, brain_area):
    if brain_area <= 0:
        raise ValueError("Brain area must be greater than zero.")
    return (tumor_area / brain_area) * 100


def experimental_severity_priority(tumor_percentage):
    if tumor_percentage < 1:
        return "Low", "Routine"
    elif tumor_percentage < 3:
        return "Moderate", "Attention"
    elif tumor_percentage < 5:
        return "High", "High"
    else:
        return "Very High", "Urgent"
