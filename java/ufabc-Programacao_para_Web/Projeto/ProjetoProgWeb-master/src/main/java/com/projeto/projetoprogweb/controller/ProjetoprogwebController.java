package com.projeto.projetoprogweb.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.servlet.ModelAndView;

import com.projeto.projetoprogweb.model.entity.Comentario;
import com.projeto.projetoprogweb.service.ComentarioService;
import com.projeto.projetoprogweb.service.NoticiaService;
import com.projeto.projetoprogweb.service.TipoNoticiaService;

@Controller
public class ProjetoprogwebController {

	//Inje��es
	/*@Autowired
	private Calendar calendar;*/
	@Autowired
	private ComentarioService comentService;
	@Autowired
	private NoticiaService noticiaService;
	@Autowired
	private TipoNoticiaService tipoService;
	
	//- - - CONTROLLER FOR JSP PAGES - - -
	
	@RequestMapping(value = { "/jsp", "/indexjsp" })
	public ModelAndView teste() {
		return new ModelAndView("inicio");
	}
	
	@RequestMapping(value = { "/listarcomentjsp" })
	@ResponseBody
	public ModelAndView listarComent() {
		ModelAndView mv = new ModelAndView("listarcoment");
		mv.addObject("comentarios", comentService.getAll());
		return mv;
	}
	
	//- - - CONTROLLER FOR HTML PAGES - - -
	@RequestMapping({ "/", "/index" })
	public ModelAndView indexhtml() {
		return new ModelAndView("index");
	}
	
	@RequestMapping("/listagem")
	public ModelAndView listar(Model model) {
		model.addAttribute("comentario", comentService.getAll());
		return new ModelAndView("listagem");
	}
	
	@RequestMapping("/noticias")
	public ModelAndView listarnoticia(Model model) {
		model.addAttribute("noticia",noticiaService.getAll());
		return new ModelAndView("noticia");
	}
	
	@GetMapping("/comenta")
	public ModelAndView recebeInserir(Model model) {
		model.addAttribute("comentario", new Comentario());

		return new ModelAndView("comenta");
	}
	
	@PostMapping("/comenta")
	public ModelAndView salvaInserir(@ModelAttribute("comentario") Comentario coment, Model model) {
		coment.setData("21-08-2018");
		comentService.create(coment);
		return new ModelAndView("comenta");
	}
}
