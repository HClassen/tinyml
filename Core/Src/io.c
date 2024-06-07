#include "stm32f7xx_hal.h"

#ifdef HAL_UART_MODULE_ENABLED
extern UART_HandleTypeDef huart1;

int __io_putchar(char ch)
{
	HAL_UART_Transmit(&huart1, (uint8_t *)&ch, 1, HAL_MAX_DELAY);
    return ch;
}
#endif /* HAL_UART_MODULE_ENABLED */
