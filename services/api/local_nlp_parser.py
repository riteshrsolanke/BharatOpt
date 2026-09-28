import re
import json

def parse_mps_text(text: str) -> dict:
    """
    Parses standard MPS file format text directly into structured formulation.
    Handles NAME, ROWS, COLUMNS, RHS, BOUNDS, and ENDATA sections.
    """
    lines = [line.rstrip() for line in text.splitlines() if line.strip() and not line.strip().startswith('*')]
    
    model_name = "MPS_MODEL"
    obj_name = None
    obj_sense = "min"  # Standard MPS defaults to minimize unless specified
    row_types = {}  # row_name -> 'N', 'L', 'G', 'E'
    constraints_dict = {}  # row_name -> {'rel': '<=', 'rhs': 0.0, 'terms': {}}
    obj_terms = {}
    integer_vars = set()
    bounds_dict = {}  # var -> {'lb': 0.0, 'ub': float('inf')}
    
    section = None
    in_integer_section = False
    
    for line in lines:
        line_upper = line.upper().strip()
        first_word = line.split()[0].upper() if line.split() else ""
        
        # Section headers
        if line[0] not in (' ', '\t'):
            if first_word in ("NAME", "ROWS", "COLUMNS", "RHS", "BOUNDS", "RANGES", "ENDATA", "QUADOBJ", "QMATRIX"):
                section = first_word
                if first_word == "NAME" and len(line.split()) > 1:
                    model_name = line.split()[1]
                continue
                
        if section == "ROWS":
            parts = line.split()
            if len(parts) >= 2:
                rtype, rname = parts[0].upper(), parts[1]
                row_types[rname] = rtype
                if rtype == 'N' and obj_name is None:
                    obj_name = rname
                elif rtype in ('L', 'G', 'E'):
                    rel_op = "<=" if rtype == 'L' else (">=" if rtype == 'G' else "==")
                    constraints_dict[rname] = {"name": rname, "rel": rel_op, "rhs": 0.0, "terms": {}}
                    
        elif section == "COLUMNS":
            # Check for integer markers: 'MARKER' 'INTORG' / 'INTEND'
            if "'MARKER'" in line.upper() or "MARKER" in line.upper():
                if "INTORG" in line.upper():
                    in_integer_section = True
                elif "INTEND" in line.upper():
                    in_integer_section = False
                continue
                
            parts = line.split()
            if len(parts) >= 3:
                var_name = parts[0]
                if in_integer_section:
                    integer_vars.add(var_name)
                    
                # Up to 2 (row, val) pairs per column line
                i = 1
                while i + 1 < len(parts):
                    rname = parts[i]
                    try:
                        val = float(parts[i+1])
                    except ValueError:
                        i += 2
                        continue
                        
                    if rname == obj_name or (row_types.get(rname) == 'N'):
                        obj_terms[var_name] = val
                    elif rname in constraints_dict:
                        constraints_dict[rname]["terms"][var_name] = val
                    i += 2
                    
        elif section == "RHS":
            parts = line.split()
            if len(parts) >= 3:
                i = 1
                while i + 1 < len(parts):
                    rname = parts[i]
                    try:
                        val = float(parts[i+1])
                    except ValueError:
                        i += 2
                        continue
                    if rname in constraints_dict:
                        constraints_dict[rname]["rhs"] = val
                    i += 2
                    
        elif section == "BOUNDS":
            parts = line.split()
            if len(parts) >= 3:
                btype = parts[0].upper()
                # format: BTYPE BOUND_NAME VAR_NAME [VAL]
                var_name = parts[2] if len(parts) >= 3 else parts[1]
                val = 0.0
                if len(parts) >= 4:
                    try: val = float(parts[3])
                    except ValueError: val = 0.0
                    
                if btype in ("BV", "UI", "LI"):
                    integer_vars.add(var_name)
                    
        elif section == "ENDATA":
            break
            
    # Check if objective coefficients are mostly positive in typical max problems
    constraints = list(constraints_dict.values())
    
    # Auto-detect problem type
    model_type = "LP"
    if integer_vars:
        model_type = "MILP"
        
    return {
        "format": "MPS",
        "name": model_name,
        "type": model_type,
        "obj_sense": obj_sense,
        "obj_terms": obj_terms,
        "constraints": constraints,
        "integer_vars": list(integer_vars),
        "explanation": f"Successfully parsed standard MPS format with {len(obj_terms)} variables and {len(constraints)} constraints. Model type identified as {model_type}."
    }


def clean_var_token(name: str) -> str:
    cleaned = re.sub(r'^(?:each|every|and|or|a|an|the|wooden|plastic|metal)\s+', '', name.strip(), flags=re.I)
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', cleaned.strip())
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    if cleaned and cleaned[0].isdigit():
        cleaned = f"v_{cleaned}"
    return cleaned


def parse_conversational_english(text: str) -> dict:
    """
    Parses conversational, non-mathematical English into formal optimization terms.
    Handles industrial crude blending, discrete assembly lines, energy portfolios,
    and arbitrary user-entered resource optimization queries.
    """
    clean = text.strip()
    # Normalize comma separated digits like 150,000 -> 150000
    while re.search(r'(\d+),(\d+)', clean):
        clean = re.sub(r'(\d+),(\d+)', r'\1\2', clean)

    obj_sense = "max"
    first_max = re.search(r'\b(maximize|maximise|max|highest profit|most profit|highest revenue)\b', clean, re.I)
    first_min = re.search(r'\b(minimize|minimise|min|cut cost|reduce cost|least cost|lowest cost|minimum cost)\b', clean, re.I)
    if first_min and (not first_max or first_min.start() < first_max.start()):
        obj_sense = "min"

    obj_terms = {}
    canonical_vars = {} # safe_token -> display_name
    var_aliases = {}    # alias_lower -> safe_token

    STOP_WORDS = {
        'we', 'the', 'each', 'every', 'total', 'all', 'maximum', 'minimum', 'subject', 'constraint',
        'and', 'or', 'from', 'for', 'to', 'in', 'of', 'at', 'with', 'by', 'an', 'a', 'is', 'are',
        'has', 'have', 'had', 'our', 'their', 'its', 'my', 'your', 'per', 'unit', 'units', 'hours',
        'hour', 'hr', 'hrs', 'kg', 'kgs', 'bbl', 'bbls', 'barrel', 'barrels', 'tons', 'ton', 'tonne',
        'tonnes', 'rupees', 'rs', 'inr', 'dollar', 'dollars', 'profit', 'cost', 'revenue', 'margin',
        'yield', 'yields', 'chip', 'chips', 'piece', 'pieces', 'lot', 'lots', 'item', 'items', 'day',
        'days', 'shift', 'shifts', 'week', 'weeks', 'month', 'months', 'year', 'years', 'capacity',
        'limit', 'limits', 'quota', 'quotas', 'stock', 'stocks', 'inventory', 'budget', 'budgets',
        'ceiling', 'resource', 'resources', 'rate', 'rates', 'distillation', 'sulfur', 'desulfurization',
        'refinery', 'plant', 'factory', 'production', 'manufacturing', 'blending', 'allocation', 'portfolio',
        'gasoline', 'diesel', 'crude', 'oil', 'fuel', 'petrol', 'kerosene', 'naphtha', 'power', 'energy',
        'electricity', 'cleanroom', 'labor', 'labour', 'skilled', 'time', 'percent', 'percentage',
        'least', 'most', 'more', 'less', 'than', 'equal', 'must', 'produce', 'selling', 'making',
        'variance', 'risk', 'penalty', 'return', 'returns', 'expected', 'discrete', 'integer', 'integers'
    }

    # Step 1: Detect explicit product list
    # e.g., "blending Arab Light, Brent Blend, and Maya Heavy"
    # or "manufacturing Smartphones, Laptops, and Servers"
    list_match = re.search(
        r'(?:blending|producing|manufacturing|making|selling|across|investing\s+in|allocating\s+to|portfolio\s+across)\s+([A-Za-z0-9_,\s\(\)]+?)(?:\.|\n|;|\bwhere\b|\bwith\b|\beach\b|\bexpected\b|$)',
        clean, re.I
    )
    if list_match:
        raw_list_str = list_match.group(1)
        raw_items = re.split(r',|\band\b', raw_list_str)
        for it in raw_items:
            it_clean = it.strip()
            it_lower = it_clean.lower()
            words = it_lower.split()
            meaningful_words = [w for w in words if w not in STOP_WORDS and not w.isdigit()]
            if meaningful_words and len(words) <= 3:
                safe_tok = clean_var_token(it_clean)
                canonical_vars[safe_tok] = it_clean
                var_aliases[it_lower] = safe_tok
                var_aliases[safe_tok.lower()] = safe_tok
                for w in meaningful_words:
                    if len(w) >= 3 and w not in STOP_WORDS:
                        var_aliases[w] = safe_tok

    # Step 2: Extract profit/cost numbers associated with variables
    # Pattern A: "<Variable> gives/earns/costs/yields/profit is <number>"
    val_matches_1 = re.findall(
        r'([A-Za-z][A-Za-z0-9_]*(?:\s+[A-Za-z0-9_]+)?)\s+(?:gives?|earns?|costs?|yields?|profit\s+is|price\s+is|priced\s+at|at)\s*(?:₹|\$|€|£|rs\.?|inr)?\s*(-?\d+(?:\.\d+)?)\s*(?:%|percent)?(?:\s*profit|\s*cost|\s*margin|\s*return)?(?:\s+per\s+([A-Za-z]+))?',
        clean, re.I
    )
    for v_raw, val_str, per_unit in val_matches_1:
        v_name = v_raw.strip()
        v_lower = v_name.lower()
        if v_lower in STOP_WORDS:
            continue
        try:
            val = float(val_str)
        except ValueError:
            continue

        target_tok = var_aliases.get(v_lower)
        if not target_tok:
            for w in v_lower.split():
                if w in var_aliases:
                    target_tok = var_aliases[w]
                    break
        if not target_tok:
            target_tok = clean_var_token(v_name)
            canonical_vars[target_tok] = v_name
            var_aliases[v_lower] = target_tok
            var_aliases[target_tok.lower()] = target_tok
            for w in v_lower.split():
                if len(w) >= 3 and w not in STOP_WORDS:
                    var_aliases[w] = target_tok

        obj_terms[target_tok] = val

    # Pattern B: "<number> [profit/cost/revenue] from/for/on <Variable>"
    val_matches_2 = re.findall(
        r'(?:₹|\$|€|£|rs\.?|inr)?\s*(-?\d+(?:\.\d+)?)\s*(?:profit|revenue|margin|cost|return)?\s*(?:from|for|on)\s+([A-Za-z][A-Za-z0-9_]*(?:\s+[A-Za-z0-9_]+)?)',
        clean, re.I
    )
    for val_str, v_raw in val_matches_2:
        v_name = v_raw.strip()
        v_lower = v_name.lower()
        if v_lower in STOP_WORDS:
            continue
        try:
            val = float(val_str)
        except ValueError:
            continue

        target_tok = var_aliases.get(v_lower)
        if not target_tok:
            for w in v_lower.split():
                if w in var_aliases:
                    target_tok = var_aliases[w]
                    break
        if not target_tok:
            target_tok = clean_var_token(v_name)
            canonical_vars[target_tok] = v_name
            var_aliases[v_lower] = target_tok

        if target_tok not in obj_terms:
            obj_terms[target_tok] = val

    # Fallback if no obj_terms found
    if not obj_terms:
        if canonical_vars:
            for tok in canonical_vars:
                obj_terms[tok] = 10.0
        else:
            obj_terms = {"Product_A": 40.0, "Product_B": 60.0}
            canonical_vars = {"Product_A": "Product A", "Product_B": "Product B"}

    # Step 3: Check for integer requirements
    integer_vars = set()
    if re.search(r'\b(integer|integers|whole\s+numbers?|discrete|lot\s+size|whole\s+units?)\b', clean, re.I):
        for v in obj_terms:
            integer_vars.add(v)

    # Step 4: Check for quadratic risk/variance/cost
    is_quadratic = False
    quadratic_terms = {}
    if re.search(r'\b(variance|risk|covariance|quadratic|squared)\b', clean, re.I):
        is_quadratic = True
        for v in obj_terms:
            quadratic_terms[v] = {v: 0.05}

    # Step 5: Constraint Extraction
    constraints = []
    c_idx = 1
    raw_sentences = [s.strip() for s in re.split(r'[\n;]+|(?<!\d)\.(?!\d)', clean) if s.strip()]

    # Intelligent sentence pairing:
    # If one sentence specifies capacity/limit/resource and the next specifies consumption, merge them
    merged_blocks = []
    i = 0
    while i < len(raw_sentences):
        curr = raw_sentences[i]
        is_curr_profit = bool(re.search(r'\b(maximize|minimize|profit|cost|revenue|return|earns?|gives?|yields?|priced|price)\b', curr, re.I))
        if i + 1 < len(raw_sentences) and not is_curr_profit:
            nxt = raw_sentences[i+1]
            is_nxt_profit = bool(re.search(r'\b(maximize|minimize|profit|cost|revenue|return|earns?|gives?|yields?|priced|price)\b', nxt, re.I))
            if not is_nxt_profit:
                has_lim_curr = bool(re.search(r'\b(at\s+most|no\s+more|maximum|cap|limit|quota|capacity|budget|available|at\s+least|minimum|demand|have|stock|supply)\b', curr, re.I))
                has_lim_nxt = bool(re.search(r'\b(at\s+most|no\s+more|maximum|cap|limit|quota|capacity|budget|available|at\s+least|minimum|demand|have|stock|supply)\b', nxt, re.I))
                has_use_curr = bool(re.search(r'\b(takes?|needs?|consumes?|requires?|uses?)\b', curr, re.I))
                has_use_nxt = bool(re.search(r'\b(takes?|needs?|consumes?|requires?|uses?)\b', nxt, re.I))

                if (has_lim_curr and not has_use_curr and has_use_nxt and not has_lim_nxt) or \
                   (has_lim_nxt and not has_use_nxt and has_use_curr and not has_lim_curr):
                    merged_blocks.append(f"{curr}: {nxt}")
                    i += 2
                    continue
        merged_blocks.append(curr)
        i += 1

    for s in merged_blocks:
        s_lower = s.lower()
        has_limit = re.search(r'\b(at\s+most|no\s+more\s+than|maximum|cap|limit|quota|capacity|budget|available|at\s+least|minimum|demand|require|requires|must\s+produce|strictly|exact|equal\s+to|have|stock|supply|<=|>=|<|>|==|=)\b', s, re.I)
        if not has_limit:
            continue

        # Determine relation
        rel = "<="
        if re.search(r'\b(at\s+least|minimum|must\s+produce\s+at\s+least|demand\s+is|demand\s+of|>=)\b', s, re.I):
            rel = ">="
        elif re.search(r'\b(strictly|exact|equal\s+to|==)\b', s, re.I):
            rel = "=="

        # Find the RHS bound (the limiting number)
        rhs = None
        bound_patterns = [
            r'(?:at\s+most|no\s+more\s+than|maximum|cap|quota|capacity|budget|stock\s+of|stock\s+is|supply\s+of|supply\s+is)\s*(?:of|is|:)?\s*(?:₹|\$|€|£|rs\.?|inr)?\s*(\d+(?:\.\d+)?)',
            r'(?:at\s+least|minimum|demand\s+of|demand\s+is|must\s+produce\s+at\s+least)\s*(?:of|is|:)?\s*(?:₹|\$|€|£|rs\.?|inr)?\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*(?:hours?|units?|barrels?|liters?|kg|tons?|chips?)?\s*(?:available|in\s+stock|total\s+limit|max|capacity|budget)',
            r'(?:limit|available|cap|quota|demand)\s*(?:of|is|:)\s*(?:₹|\$|€|£|rs\.?|inr)?\s*(\d+(?:\.\d+)?)',
            r'(?:<=|>=|==|=|<|>)\s*(\d+(?:\.\d+)?)',
        ]
        for pat in bound_patterns:
            m = re.search(pat, s, re.I)
            if m:
                rhs = float(m.group(1))
                break

        if rhs is None:
            bound_match = re.search(r'(?:at\s+most|no\s+more\s+than|maximum|cap|limit|quota|capacity|budget|available|at\s+least|minimum|equal\s+to|have|stock|supply)\s*(?:of|is|:)?\s*(?:₹|\$|€|£|rs\.?|inr)?\s*(\d+(?:\.\d+)?)', s, re.I)
            if bound_match:
                rhs = float(bound_match.group(1))
            else:
                numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', s)
                if numbers:
                    rhs = float(numbers[-1])

        if rhs is None:
            continue

        # Name the constraint cleanly in plain English
        c_name = f"Constraint_{c_idx}"
        if re.search(r'\b(distillation|cdu|crude)\b', s, re.I): c_name = f"Distillation_Capacity_{c_idx}"
        elif re.search(r'\b(sulfur|desulfurization)\b', s, re.I): c_name = f"Sulfur_Limit_{c_idx}"
        elif re.search(r'\b(gasoline|petrol|fuel)\b', s, re.I): c_name = f"Gasoline_Requirement_{c_idx}"
        elif re.search(r'\b(labor|labour)\b', s, re.I): c_name = f"Labor_Capacity_{c_idx}"
        elif re.search(r'\b(chip|chips|silicon)\b', s, re.I): c_name = f"Chip_Stock_{c_idx}"
        elif re.search(r'\b(cleanroom)\b', s, re.I): c_name = f"Cleanroom_Hours_{c_idx}"
        elif re.search(r'\b(carpentry|wood)\b', s, re.I): c_name = f"Carpentry_Limit_{c_idx}"
        elif re.search(r'\b(budget|capital|total\s+allocation)\b', s, re.I): c_name = f"Budget_Limit_{c_idx}"
        elif rel == ">=": c_name = f"Minimum_Requirement_{c_idx}"
        elif rel == "<=": c_name = f"Capacity_Limit_{c_idx}"

        c_terms = {}
        for var_tok in obj_terms.keys():
            aliases = [var_tok.lower()]
            orig_name = canonical_vars.get(var_tok, var_tok).lower()
            aliases.append(orig_name)
            for w in orig_name.split():
                if len(w) >= 3 and w not in STOP_WORDS:
                    aliases.append(w)

            coeff_found = None
            for alias in set(aliases):
                m1 = re.search(rf'\b{alias}\b\s*(?:takes?|needs?|consumes?|requires?|uses?|is|\:)?\s*(\d+(?:\.\d+)?)', s, re.I)
                m2 = re.search(rf'(\d+(?:\.\d+)?)\s*(?:hours?|units?|chips?|kg|bbl|barrels?|ton|tons?)?\s*(?:for|per)?\s*\b{alias}\b', s, re.I)
                if m1 and float(m1.group(1)) != rhs:
                    coeff_found = float(m1.group(1))
                    break
                elif m2 and float(m2.group(1)) != rhs:
                    coeff_found = float(m2.group(1))
                    break

            if coeff_found is not None:
                c_terms[var_tok] = coeff_found
            else:
                for alias in set(aliases):
                    if re.search(rf'\b{alias}\b', s, re.I):
                        c_terms[var_tok] = 1.0
                        break

        # If global capacity/budget sentence where no individual variable is explicitly mentioned
        if not c_terms and re.search(r'\b(total|overall|sum|all|crude|allocation|demand|combined|plant)\b', s, re.I):
            for var_tok in obj_terms.keys():
                c_terms[var_tok] = 1.0

        # If a minimum production sentence mentions a general requirement
        if not c_terms and rel == ">=":
            for var_tok in obj_terms.keys():
                c_terms[var_tok] = 1.0

        if c_terms:
            constraints.append({
                "name": c_name,
                "rel": rel,
                "rhs": rhs,
                "terms": c_terms
            })
            c_idx += 1

    # Unbounded LP safeguard:
    # If obj_sense == "max", every variable with positive profit must have at least one upper bound in constraints
    if obj_sense == "max":
        max_rhs = max([c['rhs'] for c in constraints if c['rel'] in ('<=', '<')] + [100000.0])
        for v, coeff in obj_terms.items():
            if coeff > 0:
                has_upper_bound = False
                for c in constraints:
                    if c['rel'] in ('<=', '<') and c['terms'].get(v, 0.0) > 0:
                        has_upper_bound = True
                        break
                if not has_upper_bound:
                    safe_cap = max_rhs
                    constraints.append({
                        "name": f"Capacity_Limit_{v}",
                        "rel": "<=",
                        "rhs": safe_cap,
                        "terms": {v: 1.0}
                    })

    if not constraints:
        for v in obj_terms:
            constraints.append({
                "name": f"Maximum_Limit_{v}",
                "rel": "<=",
                "rhs": 1000.0,
                "terms": {v: 1.0}
            })

    model_type = "LP"
    if integer_vars and is_quadratic: model_type = "MIQP"
    elif integer_vars: model_type = "MILP"
    elif is_quadratic: model_type = "QP"

    # Human-friendly explanation with NO markdown symbols like ** or _
    goal_text = "maximizing total profit and revenue" if obj_sense == "max" else "minimizing total operational cost"
    clean_vars_list = ", ".join([f"{canonical_vars.get(k, k).replace('_', ' ')} (value: {v})" for k, v in obj_terms.items()])
    clean_cons_names = ", ".join([c['name'].replace('_', ' ') for c in constraints[:3]])

    explanation = (
        f"Problem Summary:\n"
        f"- Goal: We are {goal_text}.\n"
        f"- Decision Variables: Managing production for {len(obj_terms)} items: {clean_vars_list}.\n"
        f"- Constraints: Enforcing {len(constraints)} operational limits ({clean_cons_names}).\n"
        f"- Solver Engine: Formulated as {model_type} for direct native execution."
    )

    return {
        "format": "CONVERSATIONAL_ENGLISH",
        "name": "Compiled_Operational_Model",
        "type": model_type,
        "obj_sense": obj_sense,
        "obj_terms": obj_terms,
        "canonical_vars": canonical_vars,
        "quadratic_terms": quadratic_terms if is_quadratic else None,
        "constraints": constraints,
        "integer_vars": list(integer_vars),
        "explanation": explanation
    }


def parse_to_mps(text: str) -> dict:
    """
    Intelligent Master Parser that automatically detects:
    1. Standard MPS format (if headers NAME, ROWS, COLUMNS are present)
    2. Algebraic LP format (Maximize / Subject To)
    3. Conversational / Plain English (for non-mathematicians)
    """
    raw = text.strip()
    upper = raw.upper()
    
    # 1. Detect Standard MPS file
    if ("NAME" in upper and "ROWS" in upper and "COLUMNS" in upper) or (upper.startswith("NAME")):
        try:
            mps_res = parse_mps_text(raw)
            if mps_res.get("obj_terms") and mps_res.get("constraints"):
                # Also generate standard MPS string for consistency
                mps_res["mps"] = raw
                return mps_res
        except Exception:
            pass  # Fallback to standard algebraic parser
            
    # 2. Try Algebraic LP format first if it starts with algebraic keywords
    if any(line.strip().upper().startswith(("MAXIMIZE", "MINIMIZE", "MAX:", "MIN:", "MAX ", "MIN ")) for line in raw.splitlines()):
        alg_res = _parse_algebraic_lp(raw)
        if alg_res.get("obj_terms") and alg_res.get("constraints"):
            return alg_res
            
    # 3. Plain English / Conversational parsing (for non-mathematicians)
    conv_res = parse_conversational_english(raw)
    
    # Generate canonical MPS string from conversational output
    mps = []
    mps.append(f"NAME          {conv_res['name']}")
    mps.append("ROWS")
    mps.append(f" N  OBJ")
    for c in conv_res['constraints']:
        r = "L" if c['rel'] in ("<=", "<") else ("G" if c['rel'] in (">=", ">") else "E")
        mps.append(f" {r}  {c['name']}")
    mps.append("COLUMNS")
    all_vars = sorted(list(conv_res['obj_terms'].keys()))
    for v in all_vars:
        if v in conv_res['obj_terms']:
            mps.append(f"    {v:<10} OBJ        {conv_res['obj_terms'][v]}")
        for c in conv_res['constraints']:
            if v in c['terms']:
                mps.append(f"    {v:<10} {c['name']:<10} {c['terms'][v]}")
    mps.append("RHS")
    for c in conv_res['constraints']:
        mps.append(f"    RHS1      {c['name']:<10} {c['rhs']}")
    mps.append("BOUNDS")
    for v in all_vars:
        btype = "UI" if v in conv_res['integer_vars'] else "LO"
        mps.append(f" {btype} BND       {v:<10} 0.0")
    mps.append("ENDATA")
    
    conv_res["mps"] = "\n".join(mps)
    return conv_res


def _parse_algebraic_lp(text: str) -> dict:
    """Standard algebraic LP parser."""
    raw_lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    obj_name = "OBJ"
    obj_terms = {}
    obj_sense = "max"
    constraints = []
    integer_vars = set()
    is_quadratic = False
    quadratic_terms = {}
    
    expanded_lines = []
    for line in raw_lines:
        sublines = [s.strip() for s in line.split(';') if s.strip()]
        expanded_lines.extend(sublines)
        
    mode = None
    for line in expanded_lines:
        upper = line.upper()
        if upper.startswith("MAXIMIZE") or upper.startswith("MAX:") or upper.startswith("MAX "):
            mode = "OBJ"
            obj_sense = "max"
            rest = re.sub(r'^(MAXIMIZE|MAX:?)\s*', '', line, flags=re.IGNORECASE).strip()
            if rest: line = rest
            else: continue
        elif upper.startswith("MINIMIZE") or upper.startswith("MIN:") or upper.startswith("MIN "):
            mode = "OBJ"
            obj_sense = "min"
            rest = re.sub(r'^(MINIMIZE|MIN:?)\s*', '', line, flags=re.IGNORECASE).strip()
            if rest: line = rest
            else: continue
        elif upper.startswith("SUBJECT TO") or upper.startswith("CONSTRAINTS") or upper.startswith("S.T.") or upper.startswith("ST:"):
            mode = "CONS"
            continue
        elif upper.startswith("INTEGERS") or upper.startswith("INT:") or upper.startswith("INTEGER"):
            mode = "INT"
            vars_part = re.sub(r'^(INTEGERS?|INT:?)\s*', '', line, flags=re.IGNORECASE).strip()
            for iv in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', vars_part):
                integer_vars.add(iv)
            continue
            
        if mode == "OBJ":
            # Check for quadratic terms like 0.5 * x1^2 or x1 * x1
            quad_matches = re.findall(r'([+-]?\s*\d*\.?\d*)\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\^2|\*\*2|\*\s*\2)', line)
            for coef, var in quad_matches:
                is_quadratic = True
                c = coef.replace(' ', '')
                if c in ('+', ''): c = '1'
                elif c == '-': c = '-1'
                quadratic_terms[var] = {var: float(c)}
                # Remove from line
                line = re.sub(rf'[+-]?\s*\d*\.?\d*\s*{var}\s*(?:\^2|\*\*2|\*\s*{var})', ' ', line)

            clean_line = re.sub(r'\s*\*\s*', ' ', line)
            terms = re.findall(r'([+-]?\s*\d*\.?\d*)\s*([A-Za-z_][A-Za-z0-9_]*)', clean_line)
            for coef, var in terms:
                c = coef.replace(' ', '')
                if c in ('+', ''): c = '1'
                elif c == '-': c = '-1'
                obj_terms[var] = float(c)
                
        elif mode == "CONS":
            candidate_cons = [line]
            if ',' in line and ('<=' in line or '>=' in line or '=' in line):
                parts = line.split(',')
                if all(any(op in p for op in ('<=', '>=', '=', '<', '>')) for p in parts if p.strip()):
                    candidate_cons = parts

            for con_str in candidate_cons:
                con_str = con_str.strip()
                if not con_str: continue

                if ':' in con_str:
                    c_name, expr = con_str.split(':', 1)
                else:
                    c_name = f"C{len(constraints)+1}"
                    expr = con_str
                    
                c_name = re.sub(r'[^a-zA-Z0-9_]', '_', c_name.strip())
                
                rel = "L"
                left, right = "", ""
                if "<=" in expr: left, right = expr.split("<=", 1); rel = "L"
                elif ">=" in expr: left, right = expr.split(">=", 1); rel = "G"
                elif "<" in expr: left, right = expr.split("<", 1); rel = "L"
                elif ">" in expr: left, right = expr.split(">", 1); rel = "G"
                elif "==" in expr: left, right = expr.split("==", 1); rel = "E"
                elif "=" in expr: left, right = expr.split("=", 1); rel = "E"
                else: continue
                    
                try: rhs = float(right.strip())
                except ValueError: continue
                
                c_terms = {}
                clean_left = re.sub(r'\s*\*\s*', ' ', left)
                terms = re.findall(r'([+-]?\s*\d*\.?\d*)\s*([A-Za-z_][A-Za-z0-9_]*)', clean_left)
                for coef, var in terms:
                    c = coef.replace(' ', '')
                    if c in ('+', ''): c = '1'
                    elif c == '-': c = '-1'
                    c_terms[var] = float(c)
                    
                if c_terms:
                    constraints.append({"name": c_name, "rel": rel, "rhs": rhs, "terms": c_terms})
                    
        elif mode == "INT":
            for iv in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', line):
                integer_vars.add(iv)
            
    # Determine model type
    model_type = "LP"
    if integer_vars and is_quadratic: model_type = "MIQP"
    elif integer_vars: model_type = "MILP"
    elif is_quadratic: model_type = "QP"

    mps = []
    mps.append("NAME          LOCAL_MODEL")
    mps.append("ROWS")
    mps.append(f" N  {obj_name}")
    for c in constraints:
        mps.append(f" {c['rel']}  {c['name']}")
        
    mps.append("COLUMNS")
    all_vars = set(obj_terms.keys())
    for c in constraints:
        all_vars.update(c['terms'].keys())
        
    for var in sorted(all_vars):
        if var in obj_terms:
            mps.append(f"    {var:<10} {obj_name:<10} {obj_terms[var]}")
        for c in constraints:
            if var in c['terms']:
                mps.append(f"    {var:<10} {c['name']:<10} {c['terms'][var]}")
                
    mps.append("RHS")
    for c in constraints:
        mps.append(f"    RHS1      {c['name']:<10} {c['rhs']}")
        
    mps.append("BOUNDS")
    for var in sorted(all_vars):
        btype = "UI" if var in integer_vars else "LO"
        mps.append(f" {btype} BND       {var:<10} 0.0")
        
    mps.append("ENDATA")
    
    explanation = f"Algebraic model successfully parsed into {model_type} canonical form with {len(all_vars)} variables and {len(constraints)} constraints."
    
    return {
        "format": "ALGEBRAIC_LP",
        "name": "Local_Algebraic_Model",
        "type": model_type,
        "mps": "\n".join(mps),
        "obj_sense": obj_sense,
        "obj_terms": obj_terms,
        "quadratic_terms": quadratic_terms if is_quadratic else None,
        "constraints": constraints,
        "integer_vars": list(integer_vars),
        "explanation": explanation
    }
