package local.smarttherm.model;

import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.NotNull;

public record AppSettings(
        @NotNull @DecimalMin("50.0") @DecimalMax("110.0") Double cpuWarningC,
        @NotNull @DecimalMin("50.0") @DecimalMax("110.0") Double gpuWarningC,
        @NotNull Boolean notificationsEnabled) {
    public static AppSettings defaults() {
        return new AppSettings(90.0, 85.0, true);
    }
}
