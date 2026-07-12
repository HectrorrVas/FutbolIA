class TrackingFilters:
    """
    Filtros espaciales y lógicos para limpiar detecciones ruidosas y evitar
    asignar IDs a personas fuera del terreno de juego.
    """
    
    @staticmethod
    def is_inside_field(x_m: float, y_m: float) -> bool:
        """
        Determina si un punto en metros reales está dentro del campo.
        Permite un margen de tolerancia exterior de 50 cm (0.5 metros) para no perder
        jugadores parados sobre la línea de banda o fondo.
        
        Dimensiones de la cancha calibrada vertical: X=[0, 41] metros, Y=[0, 68] metros.
        """
        margin = 0.5
        return (-margin <= x_m <= 41.0 + margin) and (-margin <= y_m <= 68.0 + margin)
