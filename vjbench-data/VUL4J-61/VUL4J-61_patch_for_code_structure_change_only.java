    private Stream<MapResult> xmlXpathToMapResult(@Name("url") String url, boolean simpleMode, String path, Map<String, Object> config) throws Exception {
        config =  config == null? Collections.emptyMap():config;
        boolean failOnError = (boolean) config.getOrDefault("failOnError", true);
        List<MapResult> result = new ArrayList<>();
        try {

            FileUtils.checkReadAllowed(url);
            DocumentBuilderFactory documentBuilderFactory = DocumentBuilderFactory.newInstance();

            XPathFactory xPathFactory = XPathFactory.newInstance();
            documentBuilderFactory.setIgnoringElementContentWhitespace(true);
            documentBuilderFactory.setNamespaceAware(true);
            
            if(StringUtils.isEmpty(path) ){
                path =  "/" ;
            }
           
            XPathExpression xPathExpression = xPathFactory.newXPath().compile(path);
            documentBuilderFactory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
            InputStream inputStream = Util.openInputStream(url, (Map) config.getOrDefault( "headers", Collections.emptyMap() ) , null);
            Document doc = documentBuilderFactory.newDocumentBuilder().parse(inputStream);
            NodeList nodeList = (NodeList) xPathExpression.evaluate(doc, XPathConstants.NODESET);

            int i = 0;
            while ( i < nodeList.getLength()) {
                final Deque<Map<String, Object>> stack = new LinkedList<>();
                handleNode(stack, nodeList.item(i), simpleMode);
                int index = 0;
                while ( index < stack.size()) {
                    result.add(new MapResult(stack.pollFirst()));
                    index++;
                }

                i++;

            }
        }
        catch (FileNotFoundException e){
            if(failOnError){
                String message = e.getMessage();
                throw new FileNotFoundException(message);
            }else{
                Map<String, Object> emptyMap =Collections.emptyMap();
                MapResult mapResult = new MapResult(emptyMap);
                return Stream.of(mapResult);
            }
        }
        catch (Exception e){
            if(failOnError){
                throw new Exception(e);
            }else{
                Map<String, Object> emptyMap =Collections.emptyMap();
                MapResult mapResult = new MapResult(emptyMap);
                return Stream.of(mapResult);
            }
                
        }
        return result.stream();
    }
