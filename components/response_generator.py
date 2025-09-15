#!/usr/bin/env python3
"""
GENERADOR DE RESPUESTAS INTELIGENTES
Componente separado para generar respuestas basadas en categorías
"""

from typing import Dict, Any, Tuple

class ResponseGenerator:
    """Generador de respuestas especializadas por categoría"""
    
    def __init__(self):
        pass
    
    def generate_response(self, category: str, question: str, diagnosis: Dict[str, Any], debug_info: Dict = None) -> Tuple[str, str]:
        """
        Genera respuesta basada en categoría detectada
        
        Args:
            category: Categoría de la pregunta
            question: Pregunta original del usuario  
            diagnosis: Información del diagnóstico
            debug_info: Información de debugging
        
        Returns:
            (response_text, response_type)
        """
        
        disease = diagnosis.get('class_name', 'Enfermedad detectada')
        confidence = diagnosis.get('confidence', 0)
        
        print(f"\n🎯 GENERANDO RESPUESTA:")
        print(f"   Categoría: {category}")
        print(f"   Enfermedad: {disease}")
        print(f"   Confianza: {confidence:.1%}")
        
        if category == "causas":
            return self._generate_causas_response(question, disease, confidence)
        elif category == "causas_cultural":
            return self._generate_causas_cultural_response(question, disease, confidence)
        elif category == "urgencia":
            return self._generate_urgencia_response(question, disease, confidence)
        elif category == "economico":
            return self._generate_economico_response(question, disease, confidence)
        elif category == "cultural":
            return self._generate_cultural_response(question, disease, confidence)
        elif category == "prevencion":
            return self._generate_prevencion_response(question, disease, confidence)
        else:
            return self._generate_general_response(question, disease, confidence)
    
    def _generate_causas_response(self, question: str, disease: str, confidence: float) -> Tuple[str, str]:
        """Respuesta específica para preguntas sobre causas"""
        print("   📝 Generando respuesta de CAUSAS...")
        
        response_type = "🔍 ANÁLISIS DE CAUSAS"
        response = f"""
**ANÁLISIS: ¿Por qué apareció {disease} en tu cultivo?**
*Confianza del diagnóstico: {confidence:.1%}*

**🌧️ CONDICIONES AMBIENTALES QUE LO FAVORECIERON:**
• **Alta humedad**: >80% HR por períodos prolongados (6-8 horas)
• **Temperatura óptima**: 15-25°C ideal para *Phytophthora infestans*
• **Mojado foliar**: Rocío matutino prolongado o riego en horas inadecuadas
• **Poca ventilación**: Cultivo muy denso o invernadero mal ventilado

**🦠 ORIGEN DEL PATÓGENO:**
• **Fuentes de inóculo**: Cultivos infectados en un radio de 5-10 km
• **Dispersión**: Esporas transportadas por viento y lluvia
• **Rastrojos**: Material vegetal infectado no eliminado completamente
• **Plantas voluntarias**: Tomates silvestres que actúan como reservorio

**⚠️ FACTORES DE RIESGO EN TU MANEJO:**
• **Variedad susceptible**: Sin resistencia genética al patógeno
• **Densidad alta**: Plantas muy juntas impiden circulación de aire
• **Nutrición desequilibrada**: Exceso de nitrógeno hace plantas más vulnerables
• **Riego inadecuado**: Aspersión foliar o riego en horas de la tarde/noche

**🔬 PROCESO DE INFECCIÓN:**
1. **Esporas llegan** a hojas húmedas (viento, lluvia, herramientas)
2. **Germinación**: En 2-4 horas con humedad >90%
3. **Penetración**: A través de estomas o heridas microscópicas
4. **Desarrollo**: Micelio se extiende en 3-5 días
5. **Esporulación**: Produce nuevas esporas en 5-7 días

**🎯 CONCLUSIÓN:**
La combinación de: condiciones climáticas favorables + presencia del patógeno + prácticas de manejo inadecuadas resultó en esta infección. El control requiere romper este ciclo.
"""
        return response, response_type
    
    def _generate_causas_cultural_response(self, question: str, disease: str, confidence: float) -> Tuple[str, str]:
        """Respuesta para causas + manejo cultural"""
        print("   📝 Generando respuesta de CAUSAS + CULTURAL...")
        
        response_type = "🌱 CAUSAS + MANEJO CULTURAL"
        response = f"""
**ANÁLISIS DE CAUSAS + SOLUCIÓN CULTURAL PARA {disease}**
*Confianza: {confidence:.1%}*

**🔍 ¿POR QUÉ APARECIÓ?**
• **Humedad excesiva**: Condiciones >80% HR favorecieron al patógeno
• **Temperatura ideal**: 15-25°C permitió desarrollo rápido
• **Manejo inadecuado**: Prácticas que aumentaron riesgo de infección

**🌿 MANEJO CULTURAL INMEDIATO (SIN QUÍMICOS):**

**1. ELIMINACIÓN SANITARIA:**
• **Remover plantas muy infectadas**: Cortar y quemar/enterrar lejos (>100m)
• **Poda selectiva**: Eliminar hojas basales y material sospechoso
• **Desinfección**: Limpiar herramientas con alcohol 70% entre plantas

**2. MANEJO DEL AMBIENTE:**
• **Mejorar ventilación**: Aumentar espaciamiento a 60-80 cm entre plantas
• **Reducir humedad**: Abrir ventanas/cortinas en invernaderos
• **Cambiar riego**: Usar goteo en lugar de aspersión, regar en la mañana

**3. PRÁCTICAS PREVENTIVAS:**
• **Mulching**: Cubrir suelo con plástico para evitar salpicaduras
• **Rotación de aire**: Usar ventiladores si es invernadero
• **Manejo nutricional**: Reducir nitrógeno, aumentar potasio y calcio

**4. MONITOREO CULTURAL:**
• **Inspección diaria**: Revisar plantas en horas de la mañana
• **Eliminación continua**: Remover material sospechoso inmediatamente
• **Control de malezas**: Eliminar solanáceas silvestres cercanas

**🛡️ PREVENCIÓN A LARGO PLAZO:**
• **Variedades resistentes**: Para próximas siembras
• **Rotación de cultivos**: 2-3 años con gramíneas o leguminosas
• **Mejora del suelo**: Incorporar materia orgánica para mejor drenaje

**✅ VENTAJAS DEL MANEJO CULTURAL:**
• Sin residuos químicos en frutos
• Menor costo económico
• Sostenible ambientalmente
• No genera resistencia del patógeno
"""
        return response, response_type
    
    def _generate_urgencia_response(self, question: str, disease: str, confidence: float) -> Tuple[str, str]:
        """Respuesta para evaluación de urgencia"""
        print("   📝 Generando respuesta de URGENCIA...")
        
        response_type = "🚨 EVALUACIÓN DE URGENCIA"
        response = f"""
**NIVEL DE URGENCIA: {disease}**
*Diagnóstico: {confidence:.1%} de confianza*

**🚨 URGENCIA: MUY ALTA - ACTUAR EN 24-48 HORAS MÁXIMO**

**⏰ ¿POR QUÉ ES TAN URGENTE?**
• **Propagación exponencial**: Cada planta infectada puede contaminar 50+ plantas en 7 días
• **Condiciones actuales**: Si hay humedad >75%, la enfermedad se expande diariamente
• **Pérdidas económicas**: Puede destruir 80-100% del cultivo en 2-3 semanas
• **Contaminación regional**: Infecta cultivos vecinos en un radio de varios kilómetros

**📅 CRONOGRAMA DE ACCIÓN CRÍTICA:**

**PRÓXIMAS 6-12 HORAS (EMERGENCIA):**
• **Identificar foco principal**: Localizar planta con más síntomas
• **Aislamiento inmediato**: Cubrir con plástico antes de remover
• **Eliminar fuente**: Cortar y destruir plantas severamente infectadas

**12-24 HORAS (CRÍTICO):**
• **Aplicación de emergencia**: Fungicida sistémico de alta eficacia
• **Tratamiento preventivo**: Aplicar en plantas sanas circundantes
• **Suspender riego foliar**: Solo goteo hasta tener control

**24-48 HORAS (URGENTE):**
• **Segunda aplicación**: Producto con diferente modo de acción
• **Poda sanitaria masiva**: Eliminar todo material sospechoso
• **Mejorar ventilación**: Maximizar circulación de aire

**48-72 HORAS (IMPORTANTE):**
• **Evaluación de eficacia**: Verificar que no aparezcan nuevas lesiones
• **Ajuste de estrategia**: Modificar frecuencia según respuesta
• **Preparar siguiente aplicación**: Mantener presión de control

**⚠️ SEÑALES DE ALERTA MÁXIMA:**
• **Esporulación blanca visible**: Aplicar el mismo día
• **Nuevas lesiones diarias**: Aumentar frecuencia de tratamientos
• **Síntomas en frutos**: Cosechar frutos sanos INMEDIATAMENTE

**🎯 ESCALA DE TIEMPO:**
• **Sin acción**: Pérdida total en 10-14 días
• **Acción tardía (>48h)**: Control difícil, pérdidas 40-60%
• **Acción inmediata (<24h)**: Control efectivo, pérdidas <20%

**💡 RECUERDA:** El tizón tardío es la enfermedad #1 más destructiva en tomate. NO HAY TIEMPO QUE PERDER.
"""
        return response, response_type
    
    def _generate_economico_response(self, question: str, disease: str, confidence: float) -> Tuple[str, str]:
        """Respuesta para opciones económicas"""
        print("   📝 Generando respuesta ECONÓMICA...")
        
        response_type = "💰 SOLUCIÓN ECONÓMICA"
        response = f"""
**TRATAMIENTO ECONÓMICO PARA {disease}**
*Confianza: {confidence:.1%}*

**💡 ESTRATEGIA COSTO-EFECTIVA COMPROBADA:**

**🧪 PRODUCTOS ECONÓMICOS QUE SÍ FUNCIONAN:**
• **Mancozeb 80% WP**: $15-25/kg - Usar 2 kg/ha cada 7 días (protectante)
• **Hidróxido de cobre**: $8-15/kg - 2.5 kg/ha preventivo en clima seco
• **Oxicloruro de cobre + Mancozeb**: $20-30/kg - Mezcla lista 1.5 kg/ha

**💰 MEZCLAS DE TANQUE (AHORRO 40-50%):**
• **Cobre + Mancozeb casero**: $12/kg vs $35/kg productos comerciales
• **Adherente económico**: Jabón neutro 0.1% en lugar de adherentes caros
• **Penetrante casero**: Aceite vegetal 0.5% mejora eficacia

**📊 ANÁLISIS DE COSTOS REAL:**

**ESTRATEGIA PREMIUM (CARA):**
• Metalaxil + Mancozeb: $85/kg × 2.5 kg = $212/ha
• Aplicaciones cada 7 días × 8 veces = $1,696/ha por ciclo
• **Total ciclo completo: $1,800-2,200/ha**

**ESTRATEGIA ECONÓMICA (NUESTRA):**
• Mancozeb preventivo: $20/kg × 2 kg × 6 veces = $240/ha
• Cobre de rescate: $12/kg × 2.5 kg × 2 veces = $60/ha
• **Total ciclo completo: $350-450/ha**

**💡 AHORRO: 70-75% con la misma eficacia**

**🌿 MEDIDAS CULTURALES (COSTO $0):**
• **Poda sanitaria**: Elimina 60-70% de inóculo sin químicos
• **Mejora de drenaje**: Reduce reinfecciones en 50%
• **Riego matutino**: Disminuye presión de enfermedad 40%
• **Espaciamiento**: Mejor ventilación = menos aplicaciones

**⏰ TIMING INTELIGENTE PARA AHORRAR:**
• **Aplicar preventivo barato**: Cuando HR >70% por 2 días consecutivos
• **Sistémico solo en emergencia**: Cuando ya hay síntomas visibles
• **Monitoreo climático**: Evita aplicaciones innecesarias

**🎯 PLAN ECONÓMICO ESPECÍFICO:**

**SEMANAS 1-4 (PREVENTIVO):**
• Cobre cada 10 días: $8/aplicación
• **Costo mensual: $24/ha**

**SEMANAS 5-8 (REFUERZO):**
• Cobre + Mancozeb cada 7 días: $15/aplicación  
• **Costo mensual: $60/ha**

**EMERGENCIA (si aparece):**
• 1 aplicación sistémica: $85/ha
• Continuar con económicos: $15/aplicación

**💰 RESULTADO FINAL:**
• **Presupuesto total**: $300-400/ha por ciclo completo
• **Efectividad**: 90-95% igual que productos caros
• **ROI**: Ahorro de $1,200-1,500/ha permite mayor rentabilidad
"""
        return response, response_type
    
    def _generate_cultural_response(self, question: str, disease: str, confidence: float) -> Tuple[str, str]:
        """Respuesta para manejo cultural"""
        print("   📝 Generando respuesta CULTURAL...")
        
        response_type = "🌱 MANEJO CULTURAL"
        response = f"""
**MANEJO CULTURAL INTEGRAL PARA {disease}**
*Confianza: {confidence:.1%}*

**🌿 ESTRATEGIA 100% CULTURAL - SIN QUÍMICOS:**

**1. ELIMINACIÓN SANITARIA INMEDIATA:**
• **Identificar plantas infectadas**: Buscar lesiones con halo amarillo
• **Corte correcto**: Usar tijera desinfectada, cortar 20cm debajo del síntoma
• **Destrucción**: Quemar o enterrar a >100m del cultivo (NO compostar)
• **Desinfección**: Alcohol 70% en herramientas entre cada planta

**2. MANEJO DEL MICROCLIMA:**
• **Espaciamiento amplio**: 60-80cm entre plantas (vs 40cm normal)
• **Poda de ventilación**: Eliminar hojas basales hasta 40cm del suelo
• **Orientación**: Hileras con vientos dominantes para mejor circulación
• **Tutorado abierto**: Evitar follaje muy denso en la parte superior

**3. MANEJO DEL AGUA:**
• **Riego por goteo exclusivo**: Eliminar completamente aspersión foliar
• **Horario matutino**: 6-8 AM para secado rápido del rocío
• **Mulching impermeable**: Plástico negro para evitar salpicaduras
• **Drenaje mejorado**: Surcos altos para evacuar agua rápidamente

**4. CONTROL BIOLÓGICO NATURAL:**
• **Trichoderma spp.**: 2-5 kg/ha al trasplante y cada 30 días
• **Bacillus subtilis**: Aplicar cada 15 días en horas frescas
• **Extractos vegetales**: Ajo + canela como repelente natural
• **Microorganismos benéficos**: Compost de calidad aporta antagonistas

**5. MANEJO NUTRICIONAL:**
• **Reducir nitrógeno**: Evita tejidos tiernos susceptibles
• **Aumentar potasio**: 150-200 kg K₂O/ha fortalece paredes celulares
• **Calcio foliar**: Aspersiones tempranas mejoran resistencia
• **Silicio**: 100-150 ppm crea barrera física en epidermis

**6. PRÁCTICAS PREVENTIVAS:**
• **Rotación estricta**: 3 años mínimo con gramíneas o leguminosas
• **Eliminación de hospedantes**: Control de solanáceas silvestres
• **Barreras físicas**: Mallas antipulgón reducen dispersión de esporas
• **Desinfección de infraestructura**: Limpiar invernaderos entre ciclos

**7. MONITOREO INTENSIVO:**
• **Inspección diaria**: Recorrer 100% del cultivo cada mañana
• **Trampas de esporas**: Detectar presencia antes de síntomas
• **Registro climático**: Anotar HR, temperatura y mojado foliar
• **Evaluación semanal**: Documentar progreso y ajustar manejo

**🔬 FUNDAMENTO CIENTÍFICO:**
• **Phytophthora infestans** necesita agua libre >6 horas para infectar
• **85% de infecciones** ocurren con HR >90% + 15-25°C
• **Eliminación sanitaria** reduce inóculo inicial en 70-80%
• **Manejo del microclima** disminuye condiciones favorables en 60%

**📈 EFECTIVIDAD ESPERADA:**
• **Control total**: 60-80% (vs 90-95% con químicos)
• **Combinado con resistencia varietal**: 80-90%
• **En condiciones de baja presión**: 90-95%
• **Beneficio adicional**: Residuo cero en frutos

**⚠️ LIMITACIONES:**
• **Requiere dedicación diaria**: No es "automático"
• **Efectividad climática**: Menos eficaz en años muy húmedos
• **Costo de mano de obra**: Más intensivo en trabajo
• **Curva de aprendizaje**: Requiere capacitación del personal

**✅ CUÁNDO ES MÁS EFECTIVO:**
• Variedades con resistencia parcial
• Invernaderos con buen control ambiental
• Productores orgánicos certificados
• Zonas con baja presión histórica de la enfermedad
"""
        return response, response_type
    
    def _generate_prevencion_response(self, question: str, disease: str, confidence: float) -> Tuple[str, str]:
        """Respuesta para prevención futura"""
        print("   📝 Generando respuesta de PREVENCIÓN...")
        
        response_type = "🛡️ PREVENCIÓN FUTURA"
        response = f"""
**PLAN DE PREVENCIÓN INTEGRAL PARA {disease}**
*Basado en diagnóstico con {confidence:.1%} de confianza*

**🌱 ESTRATEGIA PREVENTIVA A LARGO PLAZO:**

**1. SELECCIÓN DE VARIEDADES RESISTENTES:**
• **Altamente resistentes**: Mountain Fresh Plus, Phoenix, Celebrity
• **Resistencia intermedia**: Iron Lady, Defiant PhR, Mountain Pride
• **Nuevas variedades**: Consultar catálogos 2024-2025 para mejores opciones
• **Portainjertos resistentes**: Para cultivos injertados

**2. ROTACIÓN Y PREPARACIÓN DEL SUELO:**
• **Rotación mínima**: 3 años con cultivos no solanáceos
• **Cultivos recomendados**: Maíz, frijol, brócoli, lechuga
• **Preparación**: Incorporar 20-30 ton/ha de materia orgánica
• **Drenaje**: Construir camellones de 20-30 cm de altura

**3. DISEÑO DEL CULTIVO PREVENTIVO:**
• **Orientación**: Hileras norte-sur para mejor ventilación
• **Espaciamiento amplio**: 50-60 cm entre plantas, 1.5-2m entre hileras
• **Altura de plantación**: Camellones elevados para drenaje
• **Barreras cortavientos**: Reducir dispersión de esporas

**4. SISTEMA DE MONITOREO PREVENTIVO:**
• **Estación meteorológica**: Control de HR, temperatura, mojado foliar
• **Modelos predictivos**: Usar TOMCAST o Blitecast para timing
• **Trampas de esporas**: Detectar presencia antes de síntomas
• **Alertas tempranas**: Sistema de notificación cuando HR >80%

**5. PROGRAMA DE APLICACIONES PREVENTIVAS:**

**CALENDARIO BASE:**
• **Semana 1-2**: Aplicación de cobre preventivo
• **Semana 3-4**: Trichoderma + Bacillus subtilis
• **Semana 5-6**: Cobre + adherente según clima
• **Semana 7+**: Programa según condiciones y modelo predictivo

**TRIGGER POINTS (PUNTOS DE ACTIVACIÓN):**
• **HR >75% por 2 días**: Aplicar preventivo inmediatamente
• **Lluvia pronosticada**: Aplicar 24-48h antes
• **Temperatura 15-25°C + humedad**: Máxima alerta
• **Reportes en la zona**: Intensificar monitoreo

**6. MANEJO DEL MICROCLIMA:**
• **Ventilación forzada**: Ventiladores en invernaderos
• **Manejo de cortinas**: Automatización según HR
• **Riego automatizado**: Solo goteo, temprano en la mañana
• **Sensores**: Monitoreo continuo de condiciones

**7. CAPACITACIÓN Y PROTOCOLOS:**
• **Personal entrenado**: Reconocimiento temprano de síntomas
• **Protocolos escritos**: Procedimientos claros de sanitización
• **Registros**: Documentar todas las aplicaciones y observaciones
• **Evaluación continua**: Ajustar programa según resultados

**🎯 PLAN ESPECÍFICO POR TEMPORADA:**

**TEMPORADA SECA (Nov-Abr):**
• Aplicaciones cada 15-20 días
• Enfoque en resistencia sistémica
• Preparación para temporada húmeda

**TEMPORADA HÚMEDA (May-Oct):**
• Aplicaciones cada 5-7 días
• Máximo control ambiental
• Monitoreo diario intensivo

**TRANSICIONES:**
• Abril-Mayo: Intensificar gradualmente
• Octubre-Noviembre: Reducir frecuencia progresivamente

**📊 INVERSIÓN VS RETORNO:**

**COSTO PREVENCIÓN ANUAL:**
• Variedades resistentes: +$50-100/ha
• Infraestructura mejorada: $200-500/ha (amortizable)
• Monitoreo y aplicaciones: $300-500/ha
• **Total: $550-1,100/ha**

**AHORRO POR EVITAR BROTES:**
• Tratamientos curativos evitados: $800-1,500/ha
• Pérdidas de rendimiento evitadas: $2,000-5,000/ha
• **ROI: 300-500% de retorno**

**🏆 OBJETIVO FINAL:**
Reducir la incidencia de tizón tardío de 60-80% (sin prevención) a menos del 10% con programa preventivo integral.

**✅ INDICADORES DE ÉXITO:**
• <5% de plantas con síntomas por ciclo
• Reducción de 70% en aplicaciones curativas
• Aumento de 20-30% en rendimiento
• Mejora de calidad de frutos
"""
        return response, response_type
    
    def _generate_general_response(self, question: str, disease: str, confidence: float) -> Tuple[str, str]:
        """Respuesta general para preguntas no categorizadas"""
        print("   📝 Generando respuesta GENERAL...")
        
        response_type = "📚 INFORMACIÓN INTEGRAL"
        response = f"""
**INFORMACIÓN COMPLETA SOBRE {disease}**
*Confianza del diagnóstico: {confidence:.1%}*

**Tu pregunta específica**: "{question}"

**🔬 ANÁLISIS DE TU CONSULTA:**
Basándome en el diagnóstico confirmado, aquí tienes la información más relevante para tu situación particular.

**📋 PLAN DE MANEJO INTEGRAL:**

**1. CONTROL INMEDIATO:**
• **Identificación**: Confirmar síntomas (lesiones con halo amarillo)
• **Aislamiento**: Tratar plantas infectadas como foco de contagio
• **Aplicación urgente**: Fungicida sistémico + contacto en primeras 24h
• **Eliminación**: Remover plantas severamente afectadas

**2. CONTROL QUÍMICO SISTEMÁTICO:**
• **Productos sistémicos**: Metalaxil, Dimetomorf, Mandipropamida
• **Productos de contacto**: Mancozeb, Clorotalonil, Copper
• **Frecuencia**: Cada 5-7 días en condiciones húmedas
• **Rotación**: Alternar grupos químicos para evitar resistencia

**3. MANEJO CULTURAL INTEGRADO:**
• **Ventilación**: Mejorar circulación de aire entre plantas
• **Riego**: Cambiar a goteo, evitar aspersión foliar
• **Poda**: Eliminar hojas basales para mejor aireación
• **Sanidad**: Desinfectar herramientas entre plantas

**4. MONITOREO Y SEGUIMIENTO:**
• **Inspección diaria**: Especialmente en horas de la mañana
• **Evaluación de eficacia**: Verificar control a los 7 días post-aplicación
• **Ajuste de programa**: Modificar frecuencia según condiciones climáticas
• **Documentación**: Registrar aplicaciones y observaciones

**5. PREVENCIÓN FUTURA:**
• **Variedades resistentes**: Considerar para próximos ciclos
• **Rotación de cultivos**: Planificar 2-3 años con no-solanáceas
• **Mejoras estructurales**: Drenaje y ventilación del cultivo
• **Programa preventivo**: Calendario de aplicaciones según clima

**🌡️ FACTORES CLIMÁTICOS CLAVE:**
• **Temperatura crítica**: 15-25°C favorece desarrollo
• **Humedad crítica**: >80% HR por >6 horas permite infección
• **Mojado foliar**: Rocío o lluvia son especialmente peligrosos
• **Viento**: Dispersa esporas hasta 10 km de distancia

**⚠️ NIVELES DE URGENCIA:**
• **CRÍTICO**: Síntomas en >25% del cultivo
• **ALTO**: Síntomas en 10-25% del cultivo  
• **MODERADO**: Síntomas en 5-10% del cultivo
• **BAJO**: Síntomas en <5% del cultivo

**💡 RECOMENDACIONES ESPECÍFICAS:**
Considerando tu pregunta particular, el enfoque debe ser integral combinando control químico inmediato, mejoras culturales y planificación preventiva para futuros ciclos.

**📞 CUÁNDO BUSCAR AYUDA ESPECIALIZADA:**
• Si los síntomas continúan expandiéndose después de 2 aplicaciones
• Si aparecen resistencias a los fungicidas utilizados
• Para diseñar un programa preventivo específico para tu zona
• Si necesitas certificación orgánica o manejo sostenible
"""
        return response, response_type